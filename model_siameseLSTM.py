import random

import itertools
import numpy as np
import tensorflow as tf

from read_data import DataSet
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import dynamic_rnn

from mytensorflow import get_initializer
from rnn import get_last_relevant_rnn_output, get_sequence_length
from nn import multi_conv1d, highway_network
from rnn_cell import SwitchableDropoutWrapper


def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            if gpu_idx > 0:
                tf.get_variable_scope().reuse_variables()
            model = Model(config, scope, rep=gpu_idx == 0)
            models.append(model)
    return models

class Model(object):
    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        # Define forward inputs here
        N, JX, VW, VC, W = \
            config.batch_size, config.max_sent_size, \
            config.word_vocab_size, config.char_vocab_size, config.max_word_size
        self.x = tf.placeholder('int32', [N, None], name='x')
        self.cx = tf.placeholder('int32', [N, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [N, None], name='x_mask')
        self.x_length = tf.placeholder('int32', [N], name='x_length')

        self.y = tf.placeholder('int32', [N, None], name='y')
        self.cy = tf.placeholder('int32', [N, None, W], name='cy')
        self.y_mask = tf.placeholder('bool', [N, None], name='y_mask')
        self.y_length = tf.placeholder('int32', [N], name='y_length')

        self.z = tf.placeholder('float32', [N, 2], name='z')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')        

        # Define misc
        self.tensor_dict = {}
        self.h_dim = config.hidden_size

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None
        self.na_prob = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()
        self.var_ema = None
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.summary.merge_all()
        self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))
    
    def _encoder(self, input_seq, input_seq_length, name="", reuse=False):        
        with tf.variable_scope("Encoder") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            cell_fw = BasicLSTMCell(self.h_dim, state_is_tuple=True, reuse=reuse)
            cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob = self.config.input_keep_prob)

            cell_bw = BasicLSTMCell(self.h_dim, state_is_tuple=True, reuse=reuse)
            cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob = self.config.input_keep_prob)

            (encoder_outputs, encoder_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                cell_bw, 
                inputs=input_seq,
                sequence_length=input_seq_length,
                dtype=tf.float32,
                scope='enc')

            # Join outputs since we are using a bidirectional RNN
            encoder_outputs = tf.concat(encoder_outputs, 2)

            if isinstance(encoder_state[0], LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_state[0].c, encoder_state[1].c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_state[0].h, encoder_state[1].h), 1, name='bidirectional_concat_h')
                encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)                

        return encoder_outputs, encoder_state
    def _build_forward(self):
        config = self.config
        N, JX, VW, VC, d, W = \
            config.batch_size, config.max_sent_size, \
            config.word_vocab_size, config.char_vocab_size, \
            config.hidden_size, config.max_word_size        
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size


        # Getting word vector

        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, JX, W, dc]
                    Acy = tf.nn.embedding_lookup(char_emb_mat, self.cy)  # [N, JX, W, dc]

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            yy = multi_conv1d(Acy, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                        else:
                            yy = multi_conv1d(Acy, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="yy")
                        xx = tf.reshape(xx, [-1, JX, dco])
                        yy = tf.reshape(yy, [-1, JX, dco])

            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, JX, d]
                    Ay = tf.nn.embedding_lookup(word_emb_mat, self.y)  # [N, JX, d]
                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['y'] = Ay
                if config.use_char_emb:
                    xx = tf.concat(axis=2, values=[xx, Ax])  # [N, M, JX, di]
                    yy = tf.concat(axis=2, values=[yy, Ay])  # [N, JQ, di]
                else:
                    xx = Ax
                    yy = Ay
            
            # highway network
            if config.highway:
                with tf.variable_scope("highway"):
                    xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                    tf.get_variable_scope().reuse_variables()
                    yy = highway_network(yy, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
        
        self.tensor_dict['xx'] = xx
        self.tensor_dict['yy'] = yy

        

        self.x_output, self.x_state = self._encoder(xx, self.x_length)
        self.y_output, self.y_state = self._encoder(yy, self.y_length, reuse=True) # use the same sentence encoder.

        eps = 0.0001
        def get_rnn_hidden_mean(outputs, mask, length):
            """
            outputs: [batch_size, time_step, hidden]
            mask: [batch_size, time_step]
            length: [batch_size]
            """
            x_sum_vec = tf.reduce_sum(tf.multiply(outputs, tf.tile(tf.expand_dims(tf.cast(mask, tf.float32), -1), [1, 1, self.h_dim*2])), 1) # [batch_size, h_dim*2]
            x_mean_vec = tf.div(x_sum_vec, eps+tf.cast(tf.tile(tf.expand_dims(length, 1), [1, self.h_dim*2]), tf.float32)) # [batch_size, h_dim*2]
            return x_mean_vec

        self.X = get_rnn_hidden_mean(self.x_output, self.x_mask, self.x_length) # [batch_size, h_dim*2]
        self.Y = get_rnn_hidden_mean(self.y_output, self.y_mask, self.y_length) # [batch_size, h_dim*2]

        inner_prod = tf.reduce_sum(tf.multiply(self.X, self.Y), 1)
        
        X_abs = tf.sqrt(tf.reduce_sum(tf.multiply(self.X, self.X), 1))
        Y_abs = tf.sqrt(tf.reduce_sum(tf.multiply(self.Y, self.Y), 1))
        denom = tf.multiply(X_abs, Y_abs) # denominator


        cos_sim = tf.expand_dims(tf.div(inner_prod, eps+denom), -1)


        self.h0 = tf.concat((self.X, self.Y, cos_sim), 1) # [batch_size, h_dim*4+1]
        
        self.W_pred = tf.get_variable("W_pred", shape=[self.h_dim * 4 + 1, 2])
        self.b_pred = tf.get_variable("b_pred", shape=[2])
        self.logits = tf.matmul(self.h0, self.W_pred) + self.b_pred


        print("logits:", self.logits)

    def _build_loss(self):
        config = self.config
        # self.z: [N, 2]
        losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.z))
        tf.add_to_collection('losses', losses)

        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            print('opname:', ema_var.op.name)
            print('var:', ema_var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config
        N, JX, VW, VC, d, W = \
            config.batch_size, config.max_sent_size, \
            config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for sent in batch.data['x_list']) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for sent in batch.data['x_list'])

            if sum(len(ques) for ques in batch.data['y_list']) == 0:
                new_JY = 1
            else:
                new_JY = max(len(ques) for ques in batch.data['y_list'])

            JX = min(JX, max(new_JX, new_JY))

        x = np.zeros([N, JX], dtype='int32')
        cx = np.zeros([N, JX, W], dtype='int32')
        x_mask = np.zeros([N, JX], dtype='bool')
        x_length = np.zeros([N], dtype='int32')        
        y = np.zeros([N, JX], dtype='int32')
        cy = np.zeros([N, JX, W], dtype='int32')
        y_mask = np.zeros([N, JX], dtype='bool')
        y_length = np.zeros([N], dtype='int32')
        z = np.zeros([N, 2], dtype='float32')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.x_length] = x_length
        feed_dict[self.cx] = cx
        feed_dict[self.y] = y
        feed_dict[self.cy] = cy
        feed_dict[self.y_mask] = y_mask
        feed_dict[self.y_length] = y_length
        feed_dict[self.z] = z

        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x_list']
        CX = batch.data['cx_list']

        Z = batch.data['z_list']
        for i, zi in enumerate(Z):
            z[i] = zi
        

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1
        
        # replace char data to index. 

        for i, xi in enumerate(X):
            for j, xij in enumerate(xi):
                if j == config.max_sent_size:
                    break
                each = _get_word(xij)
                assert isinstance(each, int), each
                x[i, j] = each
                x_mask[i, j] = True
            x_length[i] = len(xi)

        for i, cxi in enumerate(CX):
            for j, cxij in enumerate(cxi):
                if j == config.max_sent_size:
                    break           
                for k, cxijk in enumerate(cxij):
                    if k == config.max_word_size:
                        break
                    cx[i, j, k] = _get_char(cxijk)

        for i, qi in enumerate(batch.data['y_list']):
            for j, qij in enumerate(qi):
                if j == config.max_sent_size:
                    break
                y[i, j] = _get_word(qij)
                y_mask[i, j] = True
            y_length[i] = len(qi)

        for i, cqi in enumerate(batch.data['cy_list']):
            for j, cqij in enumerate(cqi):
                if j == config.max_sent_size:
                    break
                for k, cqijk in enumerate(cqij):
                    if k == config.max_word_size:
                        break
                    cy[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break


        return feed_dict

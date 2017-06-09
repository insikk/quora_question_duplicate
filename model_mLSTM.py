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

        with tf.variable_scope('lstm_s_premise'):
            lstm_s_premise_cell = BasicLSTMCell(self.h_dim, forget_bias=0.0)
            h_s, _ = tf.nn.dynamic_rnn(lstm_s_premise_cell, xx, sequence_length=self.x_length,
                                       dtype=tf.float32)            

        with tf.variable_scope('lstm_t_hypothesis'):
            lstm_t_hypothesis_cell = BasicLSTMCell(self.h_dim, forget_bias=0.0)
            h_t, _ = tf.nn.dynamic_rnn(lstm_t_hypothesis_cell, yy, sequence_length=self.y_length,
                                       dtype=tf.float32)


        mLSTM_cell = BasicLSTMCell(self.h_dim, forget_bias=0.0)        

        outputs = matchLSTM(
            self.h_dim,
            N,
            mLSTM_cell,
            h_t,
            h_s,
            self.y_length,
            self.x_length, 
            JX, 
            JX)        
        
        hidden_output_batch = get_last_relevant_rnn_output(outputs, self.y_length)        

        with tf.variable_scope('fully_connect'):
            _initializer = tf.truncated_normal_initializer(stddev=0.1)
            w_fc = tf.get_variable(shape=[self.h_dim, 2],
                                   initializer=_initializer, name='w_fc')
            b_fc = tf.get_variable(shape=[2],
                                   initializer=_initializer, name='b_fc')
            self.logits = tf.matmul(hidden_output_batch, w_fc) + b_fc

        print("logits:", self.logits)
        
    def _build_loss(self):
        config = self.config
        JX = tf.shape(self.x)[1]
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
        NULL = _get_word("-NULL-")
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
            y[i, len(qi)] = NULL
            y_mask[i, len(qi)] = True
            y_length[i] = len(qi)+1

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


def matchLSTM(num_units, batch_size, cell, h_t_hypothesis, h_s_premise, length_t_hypothesis, length_s_premise, max_length_t_hypothesis, max_length_s_premise):
    """
    h_s_premise: [batch_size, timestep, dim]
    h_t_hypothesis: [batch_size, timestep, dim]
    """   

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    # (B,T,D) => (T,B,D)
    h_t_hypothesis = tf.transpose(h_t_hypothesis, [1, 0, 2])
    print("h_t_hypothesis", h_t_hypothesis)

    # (B,T,D) => (T,B,D)
    h_s_premise = tf.transpose(h_s_premise, [1, 0, 2])
    print("h_s_premise", h_s_premise)

    def _match_attention(batch_size, cell, k, h_s_premise, h_t_hypothesis, length_s_premise, state, output_arr):
        """
        h_s_premise: [timestep, batch_size, dim]
        h_t_hypothesis: [timestep, batch_size, dim]
        k: iterator index
        state: [batch_size, state_tuple]
        """
        # h_t_hypothesis[k] has size of [batch_size, dim]
        h_t_k_hypothesis = h_t_hypothesis[k] # [batch_size, h_dim]


        # Actually, h_s's tensor shape is [50, 300](50 is the maximum length), which means that it contains 
        # 'pad' in the tail of the sequence. But through slice method, h_s_j's tensor shape becomes 
        # [X, 300](X is the original sentence length),which cut off the 'pad' part.

        # h_s_j = tf.slice(h_s, begin=[0, 0, 0], size=[-1, length_s, self.h_dim]) # [batch_size, X, h_dim]

        with tf.variable_scope('attention_w'):
            _initializer = tf.truncated_normal_initializer(stddev=0.1)
            w_s = tf.get_variable(shape=[num_units, num_units],
                                initializer=_initializer, name='w_s')
            w_t = tf.get_variable(shape=[num_units, num_units],
                                initializer=_initializer, name='w_t')
            w_m = tf.get_variable(shape=[num_units, num_units],
                                initializer=_initializer, name='w_m')
            w_e = tf.get_variable(shape=[num_units, 1],
                                initializer=_initializer, name='w_e')

        last_m_h = state.h # [batch_size, h_dim]

        W_s_h_s = tf.reshape(tf.matmul(tf.reshape(h_s_premise, [-1, num_units]), w_s), [-1, batch_size, num_units]) # [premise_length, batch_size, h_dim]
        
        current_timestep_factor = tf.matmul(h_t_k_hypothesis, w_t) + tf.matmul(last_m_h, w_m) # [batch_size, h_dim]

        sum_h = W_s_h_s + tf.tile(tf.expand_dims(current_timestep_factor, 0), [max_length_s_premise, 1, 1]) # [premise_length, batch_size, h_dim] 
        G_k = tf.tanh(sum_h) # [premise_length, batch_size, h_dim] 


        e_kj = tf.reshape(tf.matmul(tf.reshape(G_k, [-1, num_units]), w_e), [max_length_s_premise, batch_size]) # [premise_length, batch_size, 1]
        
        # Put mask based on length. Remove unused score exceeding its premise length. 
        mask = tf.sequence_mask(length_s_premise, max_length_s_premise, dtype=tf.float32, name='masks_for_softmax') # [batch_size, max_length_s_premise]            
        alpha_kj = tf.nn.softmax(tf.multiply(mask, tf.transpose(e_kj))) # [batch_size, premise_length]
        
        alpha_weight = tf.tile(tf.expand_dims(alpha_kj, -1), [1, 1, num_units])

        a_k = tf.reduce_sum(tf.multiply(alpha_weight, tf.transpose(h_s_premise, [1, 0, 2])), 1)
        a_k.set_shape([batch_size, num_units]) # [batch_size, h_dim]

        m_k = tf.concat([a_k, h_t_k_hypothesis], axis=1) # [batch_size, h_dim]
        with tf.variable_scope('lstm_m'):
            _, new_state = cell(inputs=m_k, state=state)


        output_arr = output_arr.write(k, new_state.h)
        k = tf.add(k, 1)        
        return k, new_state, output_arr


    state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    output_arr = tf.TensorArray(dtype=tf.float32, size=max_length_t_hypothesis)

    k = tf.constant(0)
    c = lambda iter, state, output_arr: tf.less(iter, max_length_t_hypothesis)
    b = lambda iter, state, output_arr: _match_attention(batch_size, cell, iter, h_s_premise, h_t_hypothesis, length_s_premise, state, output_arr)
    res = tf.while_loop(cond=c, body=b, loop_vars=(k, state, output_arr))

    print("output_arr:", output_arr)
    
    outputs = tf.transpose(res[-1].stack(), [1, 0, 2])    
    print("outputs:", outputs)
    
    return outputs  # [batch_size, max_length_t_hypothesis, h_dim]
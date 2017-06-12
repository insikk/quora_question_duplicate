import random

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell
from basic.attention_gru_cell import AttentionGRUCell

from basic.read_data import DataSet
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from basic.attention_gru_cell import AttentionGRUCell


def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            if gpu_idx > 0:
                tf.get_variable_scope().reuse_variables()
            model = Model(config, scope, rep=gpu_idx == 0)
            models.append(model)
    return models



def get_last_relevant_rnn_output(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant # [batch_size, out_size]

class Model(object):
    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        self.x = tf.placeholder('int32', [N, M, JX], name='x') # Make fixed size place holder for easier manipulation for Dynamic memory network.
        self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')
        self.q = tf.placeholder('int32', [N, JQ], name='q')
        self.cq = tf.placeholder('int32', [N, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
        self.y = tf.placeholder('bool', [N, None, None], name='y')        
        self.y2 = tf.placeholder('bool', [N, None, None], name='y2')
        self.wy = tf.placeholder('bool', [N, None, None], name='wy')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')

        # Define misc
        self.tensor_dict = {}

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None

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
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        print("M: ", M)
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]
                    Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    Acq = tf.reshape(Acq, [-1, JQ, W, dc])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                        xx = tf.reshape(xx, [-1, M, JX, dco])
                        qq = tf.reshape(qq, [-1, JQ, dco])

            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['q'] = Aq
                if config.use_char_emb:
                    xx = tf.concat(axis=3, values=[xx, Ax])  # [N, M, JX, di]
                    qq = tf.concat(axis=2, values=[qq, Aq])  # [N, JQ, di]
                else:
                    xx = Ax
                    qq = Aq

        embedding_dim = tf.shape(xx)[3]

        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq

        # cells for contextual embedding. 
        cell_fw = GRUCell(d)
        cell_bw = GRUCell(d)
        d_cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=config.input_keep_prob)


        cell2_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell2_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell2_fw = SwitchableDropoutWrapper(cell2_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell2_bw = SwitchableDropoutWrapper(cell2_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        cell3_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell3_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell3_fw = SwitchableDropoutWrapper(cell3_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell3_bw = SwitchableDropoutWrapper(cell3_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        cell4_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell4_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell4_fw = SwitchableDropoutWrapper(cell4_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell4_bw = SwitchableDropoutWrapper(cell4_bw, self.is_train, input_keep_prob=config.input_keep_prob)


        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("prepro"):
            # Get contextual embedding for question (query) from biRNN
            (fw_u, bw_u), _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat(axis=2, values=[fw_u, bw_u])

            # Get contextual embedding for text from biRNN
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()

                # xx is [N, M, JX, embd_size]. Convert it to [N*M, JX, embd_size]
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, tf.reshape(xx, [N*M, JX, -1]) , tf.reshape(x_len, [N*M]), dtype='float', scope='u1')  # [N, M, JX, 2d]
            else:
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, tf.reshape(xx, [N*M, JX, -1]) , tf.reshape(x_len, [N*M]), dtype='float', scope='h1')  # [N, M, JX, 2d]
            
            fw_h = tf.reshape(fw_h, [N, M, JX, -1]) # Convert from [N*M, JX, output_size] to [N, M, JX, output_size]
            bw_h = tf.reshape(fw_h, [N, M, JX, -1])

            h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):
            if config.dynamic_att:
                p0 = h
                print("M : ", M)
                u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
                q_mask = tf.reshape(tf.tile(tf.expand_dims(self.q_mask, 1), [1, M, 1]), [N * M, JQ])

                first_cell_fw = AttentionCell(cell2_fw, u, mask=q_mask, mapper='sim',
                                            input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
                first_cell_bw = AttentionCell(cell2_bw, u, mask=q_mask, mapper='sim',
                                            input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
                second_cell_fw = AttentionCell(cell3_fw, u, mask=q_mask, mapper='sim',
                                            input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
                second_cell_bw = AttentionCell(cell3_bw, u, mask=q_mask, mapper='sim',
                                               input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                print("u_shape :", u.get_shape().as_list(), N, M)
                # size of u: [N, JQ, 2d]
                g1 = self.inference(config, h, u, x_len, q_len, JX, JQ)  #[N, JX, 2*d]
                first_cell_fw = d_cell2_fw
                second_cell_fw = d_cell3_fw
                first_cell_bw = d_cell2_bw
                second_cell_bw = d_cell3_bw
            

            logits = get_logits([g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                mask=self.x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')
            a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), tf.reshape(logits, [N, M * JX]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])

            (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(d_cell4_fw, d_cell4_bw, tf.concat(axis=3, values=[g1, a1i, g1 * a1i]),
                                                          x_len, dtype='float', scope='g2')  # [N, M, JX, 2d]
            g2 = tf.concat(axis=3, values=[fw_g2, bw_g2])
            logits2 = get_logits([g2, g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                                 mask=self.x_mask,
                                 is_train=self.is_train, func=config.answer_func, scope='logits2')

            flat_logits = tf.reshape(logits, [-1, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            yp = tf.reshape(flat_yp, [-1, M, JX])
            flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            flat_yp2 = tf.nn.softmax(flat_logits2)
            yp2 = tf.reshape(flat_yp2, [-1, M, JX])
            wyp = tf.nn.sigmoid(logits2)
            print("M 2: ", M)
            self.tensor_dict['g1'] = g1
            self.tensor_dict['g2'] = g2

            self.logits = flat_logits
            self.logits2 = flat_logits2
            self.yp = yp
            self.yp2 = yp2
            self.wyp = wyp

    def _build_loss(self):
        config = self.config
        JX = tf.shape(self.x)[2]
        M = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]
        loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss)
        ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float')))
        tf.add_to_collection("losses", ce_loss2)

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

    def get_yp(self):
        return self.yp

    def get_yp2(self):
        return self.yp2

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
            JX = min(JX, new_JX)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in batch.data['x'])
            M = min(M, new_M)

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x']
        CX = batch.data['cx']

        if supervised:
            y = np.zeros([N, M, JX], dtype='bool')
            y2 = np.zeros([N, M, JX], dtype='bool')
            wy = np.zeros([N, M, JX], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2
            feed_dict[self.wy] = wy

            for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                j2, k2 = stop_idx
                if config.single:
                    X[i] = [xi[j]]
                    CX[i] = [cxi[j]]
                    j, j2 = 0, 0
                if config.squash:
                    offset = sum(map(len, xi[:j]))
                    j, k = 0, k + offset
                    offset = sum(map(len, xi[:j2]))
                    j2, k2 = 0, k2 + offset
                y[i, j, k] = True
                y2[i, j2, k2-1] = True
                if j == j2:
                    wy[i, j, k:k2] = True
                else:
                    wy[i, j, k:len(batch.data['x'][i][j])] = True
                    wy[i, j2, :k2] = True

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

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    x[i, j, k] = each
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        return feed_dict

    def memory_match(self, mem_dim, u, h, u_length, h_length, JX, JQ, N, reuse=None):
        """
        Generate episode by applying attention to current fact vectors through a modified GRU
        
        Args:
        u: [N, JQ, 2d]
        h: [N, 1, JX, 2d] # assume we use only single sentence (no split version of input)

        N: batch_size
        
        Return:            
            outputs: [N, JQ, h_dim]
        """
        h_s_premise = u # question as premise
        h_t_hypothesis = tf.squeeze(h, 1) # passage as hypothesis

        h_s_premise_length = u_length
        h_t_hypothesis_length = tf.squeeze(h_length, 1)

        mLSTM_cell = BasicLSTMCell(mem_dim, forget_bias=0.0, reuse=reuse)

        outputs = matchLSTM(
            mem_dim,
            N,
            mLSTM_cell,
            h_t_hypothesis,
            h_s_premise,
            h_t_hypothesis_length,
            h_s_premise_length,             
            JX, 
            JQ)        

        return outputs # [N, JX, mem_dim]

    def inference(self, config, h, u, h_len, u_len, JX, JQ):
        """
            Performs inference on the DMN model
        
        h:  [N, M, JX, 2d]
        u:  [N, JQ, 2d]


        return:
            output: [N, M, 2d]. 2d vector is memory for sentence + question. 
        """                    
        M = config.max_num_sents
        N = config.batch_size
        print("u : ", u.get_shape().as_list())
        print("h : ", h.get_shape())

        mem_dim = config.hidden_size * 2
        num_hops = 2

        # multiple inference
        with tf.variable_scope("multi_inference", initializer=tf.contrib.layers.xavier_initializer()):
            prev_mem_aware_question = u # initialize with u. 

            with tf.variable_scope("memory_match"):
                outputs = self.memory_match(mem_dim, prev_mem_aware_question, h, u_len, h_len, JX, JQ, N)  #[N, JX, mem_dim]
                context_repr = get_last_relevant_rnn_output(outputs, tf.reshape(h_len, [N])) # [N, mem_dim]
                print("context_repr:", context_repr)

            
            # generates n_hops episodes
            for i in range(num_hops):
                # get a new contextual representation on current memory   
                # prev_mem_aware_question may contain information about where to look on the passage                                            

                # untied weights for memory update
                with tf.variable_scope("mem_aware_question_gen_%d" % i):
                    
                    # prev_memory: memory considering the sentnece and memory. 
                    memory_update_input = tf.concat([u, prev_mem_aware_question, tf.tile(tf.expand_dims(context_repr, 1), [1, JQ, 1])], 2) # [N, JQ, 4*d + mem_dim]

                    cell_fw = GRUCell(config.hidden_size)
                    cell_bw = GRUCell(config.hidden_size)
                    d_cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=config.input_keep_prob)
                    d_cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=config.input_keep_prob)
                    (fw_u, bw_u), _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, memory_update_input, u_len, dtype='float')  # [N, J, d], [N, d]
                    prev_mem_aware_question = tf.concat(axis=2, values=[fw_u, bw_u])        

                with tf.variable_scope("memory_match"):
                    tf.get_variable_scope().reuse_variables()
                    reuse=True
                    outputs = self.memory_match(mem_dim, prev_mem_aware_question, h, u_len, h_len, JX, JQ, N, reuse)  #[N, JX, mem_dim]      
                    context_repr = get_last_relevant_rnn_output(outputs, tf.reshape(h_len, [N])) # [N, mem_dim]       
        
        return tf.expand_dims(outputs, 1) # [N, 1, JX, mem_dim]



def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "bi_attention"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
        u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
        u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
        h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
        h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(config, is_train, h, u, r, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        JX = config.max_sent_size
        M = config.max_num_sents
        JQ = config.max_ques_size
        r_expand = tf.tile(tf.expand_dims(r, 2), [1, 1, JX, 1])
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
        if not config.c2q_att:
            u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
        print("r_expand: ", r_expand.get_shape())
        print("h: ", h.get_shape())
        if config.q2c_att:
            p0 = tf.concat([h, u_a, h * u_a, h * h_a, r_expand], 3)
        else:
            p0 = tf.concat([h, u_a, h * u_a], 3)
        return p0


def matchLSTM(num_units, batch_size, cell, h_t_hypothesis, h_s_premise, length_t_hypothesis, length_s_premise, max_length_t_hypothesis, max_length_s_premise):
    """
    Modified matchLSTM. matching in light of both premise(question) and memory. 

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
        # Actually, h_s's tensor shape is [50, 300](50 is the maximum length), which means that it contains 
        # 'pad' in the tail of the sequence. But through slice method, h_s_j's tensor shape becomes 
        # [X, 300](X is the original sentence length),which cut off the 'pad' part.

        # h_s_j = tf.slice(h_s, begin=[0, 0, 0], size=[-1, length_s, self.h_dim]) # [batch_size, X, h_dim]        

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
    
    return outputs  # [batch_size, max_length_t_hypothesis, dim]
import tensorflow as tf


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

    def _match_attention(batch_size, cell, k, h_s_premise, h_t_hypothesis, length_s_premise, state, output_arr, alignment_att_arr):
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


        e_kj = tf.reshape(tf.matmul(tf.reshape(G_k, [-1, num_units]), w_e), [max_length_s_premise, batch_size]) # [premise_length, batch_size]        
        e_kj = tf.transpose(e_kj)
        
        # Put mask based on length. Remove unused score exceeding its premise length. 
        # masked softmax.
        mask = tf.sequence_mask(length_s_premise, max_length_s_premise, dtype=tf.float32, name='masks_for_softmax') # [batch_size, max_length_s_premise]        
        
        mask_not = 1-mask        
        # adhoc to remove attention in masked-out area
        alpha_kj = tf.nn.softmax(tf.multiply(mask, e_kj) + mask_not * (-10000)) # [batch_size, premise_length]                
        
        alpha_weight = tf.tile(tf.expand_dims(alpha_kj, -1), [1, 1, num_units])

        a_k = tf.reduce_sum(tf.multiply(alpha_weight, tf.transpose(h_s_premise, [1, 0, 2])), 1)
        a_k.set_shape([batch_size, num_units]) # [batch_size, h_dim]

        m_k = tf.concat([a_k, h_t_k_hypothesis], axis=1) # [batch_size, h_dim]
        with tf.variable_scope('lstm_m'):
            _, new_state = cell(inputs=m_k, state=state)


        alignment_att_arr = alignment_att_arr.write(k, alpha_kj)
        output_arr = output_arr.write(k, new_state.h)
        k = tf.add(k, 1)        
        return k, new_state, output_arr, alignment_att_arr


    state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    output_arr = tf.TensorArray(dtype=tf.float32, size=max_length_t_hypothesis)
    alignment_att_arr = tf.TensorArray(dtype=tf.float32, size=max_length_t_hypothesis)

    k = tf.constant(0)
    c = lambda iter, state, output_arr, alignment_att_arr: tf.less(iter, max_length_t_hypothesis)
    b = lambda iter, state, output_arr, alignment_att_arr: _match_attention(batch_size, cell, iter, h_s_premise, h_t_hypothesis, length_s_premise, state, output_arr, alignment_att_arr)
    res = tf.while_loop(cond=c, body=b, loop_vars=(k, state, output_arr, alignment_att_arr))

    print("output_arr:", output_arr)
    
    outputs = tf.transpose(res[-2].stack(), [1, 0, 2]) # outputs: [batch_size, max_length_t_hypothesis, h_dim]   
    alignment_att_arr = tf.transpose(res[-1].stack(), [1, 0, 2]) # [batch_size, max_length_t_hypothesis, max_premise_length]   
    print("outputs:", outputs)
    
    return outputs, alignment_att_arr  

import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense,BatchNormalization


class MLP_input_w(Layer):
    def __init__(self):
        super(MLP_input_w, self).__init__()
        self.linear_1 = Dense(units=1024, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLP_input_v(Layer):
    def __init__(self):
        super(MLP_input_v, self).__init__()
        self.linear_1 = Dense(units=1024, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock0(Layer):
    def __init__(self):
        super(MLPBlock0, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock1(Layer):
    def __init__(self):
        super(MLPBlock1, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock2(Layer):
    def __init__(self):
        super(MLPBlock2, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock3(Layer):
    def __init__(self):
        super(MLPBlock3, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


class MLPBlock4(Layer):
    def __init__(self):
        super(MLPBlock4, self).__init__()
        self.linear_1 = Dense(units=512, activation='relu')
        self.linear_2 = Dense(units=512, activation='relu')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        return x


def GNN_layer(x,v,MLP0,MLP1,MLP2,MLP3,MLP4, num_user):
    v0 = MLP0(v)
    x_out = {0:0}
    for ii in range(num_user):
        tmp = []
        for jj in range(num_user):
            if jj != ii:
                tmp.append(MLP1(x[jj]))
        # x_max = tf.reduce_max(tmp,axis=0)
        x_max = tf.reduce_mean(tmp,axis=0)
        if num_user == 1:
            x_concate = tf.concat([x[ii], v0], axis=1)
        else:
            x_concate = tf.concat([x[ii], x_max, v0], axis=1)
        x_out[ii] = MLP2(x_concate)
    tmp = []
    for ii in range(num_user):
        tmp.append(MLP3(x[ii]))
    x_max = tf.reduce_max(tmp, axis=0)
    x_concate = tf.concat([x_max, v0], axis=1)
    v_out = MLP4(x_concate)
    return x_out,v_out


class SumRateModel:
    def __init__(self, num_antenna_bs, num_elements_irs, num_user, len_pilot, input_flag, Pt):

        # ===model parameter
        # params_system = (num_antenna_bs, num_elements_irs, num_user)
        self._params_system = (num_antenna_bs, num_elements_irs, num_user)
        self._len_pilot = len_pilot
        self._Pt = 10**(Pt/10)
        self._sigma2 = 1
        self._input_flag = input_flag

        # ===graph====
        self._graph = None
        self._sess = None
        self._initializer = None
        self._saver = None

        # ====training parameter
        self._trainOp = None
        self._learning_rate = None
        self._loss = None

        # === output of the neural network
        self._output_w = None
        self._output_theta = None
        self._rate_all = None

        # # ===channels=====
        self._channel_bs_irs_user = None
        self._channel_bs_user = None
        self._input_y = None
        self._input_y_ks = None
        self._input_locations = None

    def create_network(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            #---input---------
            (num_antenna_bs, num_elements_irs, num_user) = self._params_system
            self._input_y = tf.placeholder(tf.float32, shape=(None, 2 * num_antenna_bs, self._len_pilot))
            self._input_y_ks = tf.placeholder(tf.float32, shape=(None, 2 * num_antenna_bs, num_user, self._len_pilot // num_user))
            self._input_locations = tf.placeholder(tf.float32, shape=(None, num_user, 3))
            self._channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * num_elements_irs, 2 * num_antenna_bs, num_user))
            self._channel_bs_user = tf.placeholder(tf.float32, shape=(None, 2 * num_antenna_bs, num_user))

            #--- GNN--------
            MLP_v_in, MLP_w_in = MLP_input_v(),MLP_input_w()
            if self._input_flag == 0:
                x_w = []
                for kk in range(num_user):
                    tmp = tf.reshape(self._input_y_ks[:, :, kk, :], [-1, 2 * num_antenna_bs * self._len_pilot // num_user])
                    x_w.append(MLP_w_in(tmp))
            elif self._input_flag == 1:
                x_w = []
                for kk in range(num_user):
                    tmp_y = tf.reshape(self._input_y_ks[:, :, kk, :],
                                     [-1, 2 * num_antenna_bs * self._len_pilot // num_user])
                    tmp_location = tf.reshape(self._input_locations, (-1,num_user*3))
                    tmp = tf.concat([tmp_y,tmp_location],axis=1)
                    x_w.append(MLP_w_in(tmp))
            elif self._input_flag == 2:
                x_w = {0: 0}
                for kk in range(num_user):
                    tmp_location = tf.reshape(self._input_locations, (-1,num_user*3))
                    x_w[kk] = MLP_w_in(tmp_location)

            input_theta = tf.reduce_mean(x_w,axis=0)
            x_theta = MLP_v_in(input_theta)

            MLP0_1, MLP1_1, MLP2_1, MLP3_1, MLP4_1 = MLPBlock0(), MLPBlock1(), MLPBlock2(), MLPBlock3(), MLPBlock4()
            MLP0_2, MLP1_2, MLP2_2, MLP3_2, MLP4_2 = MLPBlock0(), MLPBlock1(), MLPBlock2(), MLPBlock3(), MLPBlock4()
            # MLP0_3, MLP1_3, MLP2_3, MLP3_3, MLP4_3 = MLPBlock0(), MLPBlock1(), MLPBlock2(), MLPBlock3(), MLPBlock4()
            x_w, x_theta =  GNN_layer(x_w, x_theta, MLP0_1, MLP1_1,MLP2_1,MLP3_1,MLP4_1, num_user)
            # x_w, x_theta =  GNN_layer(x_w, x_theta, MLP0_1, MLP1_1,MLP2_1,MLP3_1,MLP4_1, num_user)
            x_w, x_theta =  GNN_layer(x_w, x_theta, MLP0_2, MLP1_2,MLP2_2,MLP3_2,MLP4_2, num_user)
            # x_w, x_theta =  GNN_layer(x_w, x_theta, MLP0_3, MLP1_3, MLP2_3, MLP3_3, MLP4_3, num_user)

            #-----theta---------
            x_theta0 = Dense(units=num_elements_irs, activation='linear')(x_theta)
            x_theta1 = Dense(units=num_elements_irs, activation='linear')(x_theta)
            theta_tmp = tf.sqrt(tf.square(x_theta0) + tf.square(x_theta1))
            theta_real = x_theta0 / theta_tmp
            theta_imag = x_theta1 / theta_tmp
            theta = tf.concat([theta_real, theta_imag], axis=1)
            theta_T = tf.reshape(theta, [-1, 1, 2 * num_elements_irs])
            self._output_theta = tf.reshape(theta, [-1, 2 * num_elements_irs, 1])
            #---------w-------
            MLP_w_out = Dense(units=2*num_antenna_bs,activation='linear')
            for kk in range(num_user):
                w_k = MLP_w_out(x_w[kk])
                w_k = tf.reshape(w_k, [-1, 2 * num_antenna_bs, 1])
                if kk == 0:
                    W = w_k
                else:
                    W = tf.concat([W,w_k],axis=2)
            W = tf.reshape(W,[-1,2*num_antenna_bs,num_user])
            W_norm = tf.norm(W, ord='euclidean', axis=(1, 2), keepdims=True)
            W_output = (W / W_norm)*tf.sqrt(self._Pt)
            self._output_w = W_output

            ####### Loss Function
            rate = []
            for k1 in range(num_user):
                A_T_k = self._channel_bs_irs_user[:, :, :, k1]
                theta_A_k_T = tf.matmul(theta_T, A_T_k)
                h_d_k = self._channel_bs_user[:, :, k1]
                h_d_k_T = tf.reshape(h_d_k, [-1, 1, 2 * num_antenna_bs])

                signal_power = []
                for k2 in range(num_user):
                    W_k = tf.reshape(W_output[:, :, k2], [-1, 2 * num_antenna_bs, 1])
                    W_real = W_k[:, 0:num_antenna_bs]
                    W_imag = W_k[:, num_antenna_bs:2 * num_antenna_bs]
                    W_mat1 = tf.concat([W_real, W_imag], axis=2)
                    W_mat2 = tf.concat([-W_imag, W_real], axis=2)
                    W_mat = tf.concat([W_mat1, W_mat2], axis=1)

                    z1 = tf.matmul(theta_A_k_T, W_mat)
                    z2 = tf.matmul(h_d_k_T, W_mat)
                    z = z1 + z2
                    z = tf.reduce_sum(tf.square(z), axis=(1, 2))
                    signal_power.append(z)

                gamma_k = signal_power[k1] / (tf.reduce_sum(signal_power, axis=0) - signal_power[k1] + self._sigma2)
                rate.append(tf.reduce_mean(tf.log(1 + gamma_k)) / tf.log(2.0))
                if k1 == 0:
                    rate_sum = tf.log(1 + gamma_k)/tf.log(2.0)
                else:
                    rate_sum = rate_sum+tf.log(1 + gamma_k)/tf.log(2.0)
            self._rate_all = tf.convert_to_tensor(rate)
            self._loss = - tf.reduce_mean(rate_sum)


    # define optimizer of the neural network
    def create_optimizer(self, training_algorithm='Adam', learning_rate=0.001, decay_rate=0.98, decay_step=300):
        with self._graph.as_default():
            # define the learning rate
            global_step = tf.Variable(0, trainable=False)
            self._learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_step,
                                                             decay_rate=decay_rate, staircase=True)
            if (training_algorithm == 0) or (training_algorithm == 'GD'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 1) or (training_algorithm == 'RMSProp'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 2) or (training_algorithm == 'Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 3) or (training_algorithm == 'AdaGrad'):
                optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
            elif (training_algorithm == 4) or (training_algorithm == 'AdaDelta'):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learning_rate)
            else:
                raise ValueError("Unknown training algorithm.")
            self._trainOp = optimizer.minimize(self._loss, global_step=global_step)


    # create initializer and session to run the network
    def create_initializer(self):
        # initializer of the neural network
        with self._graph.as_default():
            self._saver = tf.train.Saver()
            self._initializer = tf.global_variables_initializer()
        self._sess = tf.Session(graph=self._graph)


    # initialize the computation graph
    def initialize(self, initial_run=True, path=None):
        if self._initializer is not None:
            if initial_run:
                self._sess.run(self._initializer)
            else:
                self._saver.restore(self._sess, path)
        else:
            raise ValueError('Initializer has not been set.')

    def train(self, input_y, input_y_ks, input_locations, channel_bs_irs_user, channel_bs_user):
        if self._trainOp is not None:
            feed_dict = {self._input_y: input_y,
                         self._input_y_ks: input_y_ks,
                         self._input_locations: input_locations,
                         self._channel_bs_irs_user: channel_bs_irs_user,
                         self._channel_bs_user: channel_bs_user}

            self._sess.run(self._trainOp, feed_dict=feed_dict)
        else:
            raise ValueError('Training algorithm has not been set.')


    # ===Compute loss===
    def get_loss(self, input_y, input_y_ks, input_locations, channel_bs_irs_user, channel_bs_user):
        loss = -self._sess.run(self._loss, feed_dict={self._input_y: input_y,
                                                      self._input_y_ks: input_y_ks,
                                                      self._input_locations: input_locations,
                                                      self._channel_bs_irs_user: channel_bs_irs_user,
                                                      self._channel_bs_user: channel_bs_user})
        rate_all = self._sess.run(self._rate_all, feed_dict={self._input_y: input_y,
                                                             self._input_y_ks: input_y_ks,
                                                             self._input_locations: input_locations,
                                                             self._channel_bs_irs_user: channel_bs_irs_user,
                                                             self._channel_bs_user: channel_bs_user})
        return loss, rate_all

    def get_w(self, input_y, input_y_ks, input_locations):
        return self._sess.run(self._output_w, feed_dict={self._input_y: input_y,
                                                         self._input_y_ks: input_y_ks,
                                                         self._input_locations: input_locations})

    def get_theta(self, input_y, input_y_ks, input_locations):
        return self._sess.run(self._output_theta, feed_dict={self._input_y: input_y,
                                                             self._input_y_ks: input_y_ks,
                                                             self._input_locations: input_locations})

    def save_model(self, path):
        save_path = self._saver.save(self._sess, path)
        return save_path

    def learning_rate(self):
        return self._sess.run(self._learning_rate)

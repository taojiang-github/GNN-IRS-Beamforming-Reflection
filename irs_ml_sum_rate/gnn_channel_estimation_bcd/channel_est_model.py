import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import keras
# from keras.layers import Layer, Dense,BatchNormalization
# from tensorflow import keras
from tensorflow.python.keras.layers import Layer, Dense, BatchNormalization


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


def GNN_layer(x,MLP1,MLP2,num_user):
    x_out = {0:0}
    for ii in range(num_user):
        tmp = []
        for jj in range(num_user):
            if jj != ii:
                tmp.append(MLP1(x[jj]))
        # x_max = tf.reduce_max(tmp,axis=0)
        x_max = tf.reduce_mean(tmp,axis=0)
        if num_user == 1:
            x_concate = x[ii]
        else:
            x_concate = tf.concat([x[ii], x_max], axis=1)
        x_out[ii] = MLP2(x_concate)
    return x_out


class CEModel:
    def __init__(self, num_antenna_bs, num_elements_irs, num_user, len_pilot, input_flag, sigma2):

        # ===model parameter
        # params_system = (num_antenna_bs, num_elements_irs, num_user)
        self._params_system = (num_antenna_bs, num_elements_irs, num_user)
        self._len_pilot = len_pilot
        self._sigma2 = sigma2
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
        self._output_A = None
        self._output_H = None
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
            # --- GNN--------
            MLP_v_in, MLP_w_in = MLP_input_v(), MLP_input_w()
            # MLP_location = Dense(units=64,activation='relu')
            if self._input_flag == 0:
                x_w = []
                for kk in range(num_user):
                    tmp = tf.reshape(self._input_y_ks[:, :, kk, :],
                                     [-1, 2 * num_antenna_bs * self._len_pilot // num_user])
                    x_w.append(MLP_w_in(tmp))

            elif self._input_flag == 1:
                x_w = []
                for kk in range(num_user):
                    tmp_y = tf.reshape(self._input_y_ks[:, :, kk, :],
                                       [-1, 2 * num_antenna_bs * self._len_pilot // num_user])
                    # tmp_location = MLP_location(tf.reshape(self._input_locations, (-1,num_user*3)))
                    tmp_location = tf.reshape(self._input_locations, (-1, num_user * 3))
                    tmp = tf.concat([tmp_y, tmp_location], axis=1)
                    x_w.append(MLP_w_in(tmp))
            elif self._input_flag == 2:
                x_w = {0: 0}
                for kk in range(num_user):
                    tmp_location = tf.reshape(self._input_locations, (-1, num_user * 3))
                    x_w[kk] = MLP_w_in(tmp_location)

            MLP1_1, MLP2_1 =  MLPBlock1(), MLPBlock2()
            MLP1_2, MLP2_2 =  MLPBlock1(), MLPBlock2()
            x_w =  GNN_layer(x_w,  MLP1_1, MLP2_1, num_user)
            x_w =  GNN_layer(x_w,  MLP1_2, MLP2_2, num_user)

            MLP_A_out = Dense(units=2 * num_elements_irs * num_antenna_bs, activation='linear')
            MLP_H_out = Dense(units=2 * num_antenna_bs, activation='linear')
            for kk in range(num_user):
                A_k = MLP_A_out(x_w[kk])
                A_k = tf.reshape(A_k, [-1, num_elements_irs, num_antenna_bs*2])
                loss1_k = tf.reduce_mean(tf.square(A_k - self._channel_bs_irs_user[:, 0:num_elements_irs, :, kk]))

                H_k = MLP_H_out(x_w[kk])
                H_k = tf.reshape(H_k, [-1, num_antenna_bs*2])
                loss2_k = tf.reduce_mean(tf.square(H_k-self._channel_bs_user[:,:,kk]))

                H_k = tf.reshape(H_k, [-1, num_antenna_bs*2,1])
                A_k = tf.reshape(A_k, [-1, num_elements_irs, num_antenna_bs*2,1])
                if kk==0:
                    A = A_k
                    H = H_k
                    loss = loss1_k+loss2_k
                else:
                    A = tf.concat([A,A_k],axis=3)
                    H = tf.concat([H,H_k],axis=2)
                    loss= loss+loss1_k+loss2_k

            self._output_A = A
            self._output_H = H
            self._loss = loss/num_user

    # define optimizer of the neural network
    def create_optimizer(self, training_algorithm='Adam', learning_rate=0.001, decay_rate=0.98, decay_step=300):
        with self._graph.as_default():
            # define the learning rate
            global_step = tf.Variable(0, trainable=False)
            self._learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_step,
                                                             decay_rate=decay_rate, staircase=True)
            # self._learning_rate = learning_rate
            # define the appropriate optimizer to use
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

            # =================================================================
            # training and initialization operators
            self._trainOp = optimizer.minimize(self._loss, global_step=global_step)

    # create initializer and session to run the network
    def create_initializer(self):
        # initializer of the neural network
        with self._graph.as_default():
            self._saver = tf.train.Saver()
            self._initializer = tf.global_variables_initializer()
        self._sess = tf.Session(graph=self._graph)

    # =========================================================================
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
        loss = self._sess.run(self._loss, feed_dict={self._input_y: input_y,
                                                      self._input_y_ks: input_y_ks,
                                                      self._input_locations: input_locations,
                                                      self._channel_bs_irs_user: channel_bs_irs_user,
                                                      self._channel_bs_user: channel_bs_user})

        return loss

    def get_H(self, input_y, input_y_ks, input_locations):
        return self._sess.run(self._output_H, feed_dict={self._input_y: input_y,
                                                         self._input_y_ks: input_y_ks,
                                                         self._input_locations: input_locations})

    def get_A(self, input_y, input_y_ks, input_locations):
        return self._sess.run(self._output_A, feed_dict={self._input_y: input_y,
                                                             self._input_y_ks: input_y_ks,
                                                             self._input_locations: input_locations})

    def save_model(self, path):
        save_path = self._saver.save(self._sess, path)
        return save_path

    def learning_rate(self):
        return self._sess.run(self._learning_rate)

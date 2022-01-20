import tensorflow as tf
import numpy as np
import random

class Agent_Multidim:
    def __init__(self, dimx, dima, actor_num):
        self.dimx = dimx
        self.dima = dima

        actor_parmas = lambda ns, na, dev: [np.random.normal(loc=-dev, scale=0.6, size=(ns, )) for i in range(na)]
        #self.actor_weights = [actor_parmas(2, dima, 2) for i in range(actor_num)] # np.array([[-1], [-2]])
        self.actor_weights = [actor_parmas(2, dima, -1) for i in range(actor_num/4)] + [actor_parmas(2, dima, -1) for i in range(actor_num/4)] \
             + [actor_parmas(2, dima, 2) for i in range(actor_num/4)] + [actor_parmas(2, dima, 3) for i in range(actor_num/4)]

        self.states = tf.placeholder(tf.float32, [None, self.dimx])
        self.next_states = tf.placeholder(tf.float32, [None, self.dimx])
        self.us = tf.placeholder(tf.float32, [None, self.dima])
        self.rs = tf.placeholder(tf.float32, [None, 1])

        self.opt_actor_w1 = tf.Variable(tf.random_normal([dimx, 40], stddev=0.1, mean=0))
        self.opt_actor_w2 = tf.Variable(tf.random_normal([40, dima], stddev=0.01, mean=0))

        self._build_critic()
        self._build_identifier()
        optimizer = tf.train.AdamOptimizer(0.0004)
        actor_loss = self.actor_loss1 + self.actor_loss2
        self.update_target_actor = optimizer.minimize(actor_loss, var_list=[self.opt_actor_w1, self.opt_actor_w2])
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.buffer = []
        self.timestamp = []
        self.actor_logger = []
        self.loss_logger = []

    def _build_critic(self):
        W1 = tf.Variable(tf.random_normal([self.dimx + self.dima, 50]))
        # b1 = tf.Variable(tf.zeros([20]))
        W2 = tf.Variable(tf.random_normal([50, 1]))

        ema = tf.train.ExponentialMovingAverage(0.99)
        self.update_target_critic = ema.apply([W1, W2, self.opt_actor_w1, self.opt_actor_w2])
        la1 = tf.tanh(tf.matmul(self.next_states, ema.average(self.opt_actor_w1)))
        actor_intermedia = tf.tanh(tf.matmul(la1, ema.average(self.opt_actor_w2)))
        # opt_u1 = actor_intermedia[:, 0] + actor_intermedia[:, 2]
        # opt_u2 = actor_intermedia[:, 1] + actor_intermedia[:, 3]
        target_critic_input = tf.concat([self.next_states, actor_intermedia], 1)
        critic_input = tf.concat([self.states, self.us], 1)

        layer = tf.tanh(tf.matmul(critic_input, W1))
        critic = tf.matmul(layer, W2)

        layer_t = tf.tanh(tf.matmul(target_critic_input, ema.average(W1)))
        target_critic = tf.matmul(layer_t, ema.average(W2))+ self.rs
        weight_decay = tf.add_n([tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)])

        self.critic_err = tf.reduce_mean(tf.square(target_critic - critic))
        self.critic_loss = self.critic_err  + 0.01*weight_decay

        la1_ = tf.tanh(tf.matmul(self.states, self.opt_actor_w1))
        actor_intermedia_ = tf.tanh(tf.matmul(la1_, self.opt_actor_w2))
        ci = tf.concat([self.states, actor_intermedia_], 1)
        layer = tf.tanh(tf.matmul(ci, W1))
        critic_ = tf.matmul(layer, W2)
        self.actor_loss1 = tf.reduce_mean(critic_)

        optimizer = tf.train.AdamOptimizer(0.002)
        self.critic_update_op = optimizer.minimize(self.critic_loss, var_list=[W1, W2])

        self.verify_id_input = ci
        self.critic_predict = critic

    def _build_identifier(self):
        w1 = tf.Variable(tf.random_normal([self.dimx + self.dima, 40]))
        b1 = tf.Variable(tf.zeros([40]))
        w2 = tf.Variable(tf.random_normal([40, self.dimx]))
        b2 = tf.Variable(tf.zeros([self.dimx]))

        id_input = tf.concat([self.states, self.us], 1)
        layer_id = tf.nn.sigmoid(tf.matmul(id_input, w1) + b1)
        self.predict_x = tf.matmul(layer_id, w2) + b2

        self.idloss = tf.reduce_mean(tf.reduce_sum(tf.square(self.next_states - self.predict_x), reduction_indices=[1]))
        self.train_id = tf.train.AdamOptimizer(0.001).minimize(self.idloss)

        layer_st = tf.nn.sigmoid(tf.matmul(self.verify_id_input, w1)+ b1)
        opt_next_x = tf.matmul(layer_st, w2) + b2
        x = self.verify_id_input[:, :self.dimx]
        self.actor_loss2 = tf.reduce_mean(self._dl1(x, opt_next_x))

    def _dl1(self, x, x_next): # tensors
        l1 = tf.reduce_sum(tf.square(x), axis=1)
        l2 = tf.reduce_sum(tf.square(x_next), axis=1)
        return l2 - l1

    def _get_target_actor(self, state):
        wa1, wa2 = self.sess.run([self.opt_actor_w1, self.opt_actor_w2])
        # print wa
        m1 = np.tanh(np.matmul(state, wa1))
        m = np.tanh(np.matmul(m1, wa2))
        return m
        # return np.array([m[0]+m[2], m[1]+m[3]])

    def update_actor(self, idx, s):
        ea = self.get_actor_output(idx, s) - self._get_target_actor(s)
        logger = [[],[]]
        for dimn in range(self.dima):
            phi = self.phi(s,dimn) 
            self.actor_weights[idx][dimn] = self.actor_weights[idx][dimn] - 0.12 * phi / ((1 + phi.dot(phi))**2) * ea[dimn] #0.04 0.1 * phi / ((1 + phi.dot(phi))**2)
            for n in range(len(self.actor_weights)):
                logger[dimn].append(self.actor_weights[n][dimn]) # adding in log
                if n != idx:
                    e = self.actor_weights[idx][dimn] - self.actor_weights[n][dimn]
                    self.actor_weights[n][dimn] = self.actor_weights[n][dimn] + 0.01*e #0.1
        self.actor_logger.append(logger)

    def update_agent(self, batchsize):
        if len(self.buffer) > batchsize:
            batch = random.sample(self.buffer, batchsize)
        elif len(self.buffer) > 10:
            batch = self.buffer
        else:
            return
        train_inputs = np.array(batch)
        xs = train_inputs[:, :self.dimx]
        next_s = train_inputs[:, -self.dimx:]
        us = train_inputs[:, self.dimx:self.dimx + self.dima]
        rs = np.expand_dims(train_inputs[:, self.dimx + self.dima], 1)

        lc, li, _, _ = self.sess.run([self.critic_err, self.idloss, self.critic_update_op, self.train_id], feed_dict={self.states: xs, self.next_states: next_s, self.us: us, self.rs: rs})
        print lc, li
        self.loss_logger.append([lc, li])
        self.sess.run(self.update_target_actor, feed_dict={self.states: xs})
        self.sess.run(self.update_target_critic)

    def perceive(self, x, u, r, next_x, t):
        unit = np.hstack([x, u, r, next_x])

        self.timestamp.append(t)
        self.buffer.append(unit)

    def get_actor_onedim(self, x, actor_n, dim_n):
        return self.actor_weights[actor_n][dim_n].dot(self.phi(x, dim_n))

    def get_actor_output(self, n, x):
        u = [self.get_actor_onedim(x, n, i) for i in range(self.dima)]
        return np.array(u)

    def Q(self, s, a):
        state = np.array([s]) if len(s.shape) != 2 else s
        action = np.array([a]) if len(a.shape) != 2 else a
        Qsa = self.sess.run(self.critic_predict, feed_dict={self.states: state, self.us: action})
        return Qsa[0, 0]

    def identifier_predict(self, s, a):
        state = np.array([s]) if len(s.shape) != 2 else s
        action = np.array([a]) if len(a.shape) != 2 else a
        return self.sess.run(self.predict_x, feed_dict={self.states: state, self.us: action})[0]

    def phi(self, x, actor_idx):
        s1 = x[actor_idx]
        s2 = x[actor_idx + self.dima]
        return 2*np.tanh(0.5*np.array([s1, s2]))

    def save_iodata(self, path):
        dat = np.array(self.buffer)
        np.savez(path + '/ctm.npz', t=np.array(self.timestamp), record=dat, actor=self.actor_logger, losses=self.loss_logger)


class Agent_RL:
    def __init__(self, dimx, dima):
        self.Qweights = np.diag(np.ones([dimx + dima], np.float32))
        self.actor_weights = [np.array([-0.4, -0.2]) for i in range(dima)]
        self.opt_k = None
        self.buffer = []
        self.timestamp = []
        self.actor_logger = []
        self.loss_logger = []
        self.dimx = dimx
        self.dima = dima

    def update_actor(self, s):
        if self.opt_k is not None:
            ea = self.get_actor_output(s) - np.matmul(self.opt_k, s)
            for dimn in range(self.dima):
                self.actor_weights[dimn] = self.actor_weights[dimn] - 0.05*self.phi(s, dimn) * ea[dimn] #0.32

    def update_critic(self, max_batchsize):
        if len(self.buffer) > max_batchsize:
            batch = random.sample(self.buffer, max_batchsize)
        elif len(self.buffer) > 16:
            batch = self.buffer
        else:
            return
        train_inputs = np.array(batch)
        xs = train_inputs[:, :self.dimx]
        next_s = train_inputs[:, -self.dimx:]
        us = train_inputs[:, self.dimx:self.dimx + self.dima]
        rs = train_inputs[:, self.dimx + self.dima]

        Quu = self.Qweights[self.dimx:, self.dimx:]
        Qux = self.Qweights[self.dimx:, :self.dimx]
        self.opt_k = np.matmul(-np.linalg.inv(Quu), Qux)
        umin = np.matmul(next_s, self.opt_k.T)

        target = rs + self.Q(next_s, umin)
        ci = lambda x, u: np.kron(np.concatenate([x, u]), np.concatenate([x, u]))
        delta_c = np.array([ci(xs[i], us[i]) for i in range(train_inputs.shape[0])])
        Qw = np.matmul(np.linalg.pinv(delta_c), target)
        self.Qweights = np.reshape(Qw, [self.dimx + self.dima, self.dimx + self.dima])

    def perceive(self, x, u, r, next_x, t):
        unit = np.hstack([x, u, r, next_x])
        self.timestamp.append(t)
        self.actor_logger.append([self.actor_weights[0], self.actor_weights[1]])
        self.buffer.append(unit)

    def get_actor_onedim(self, x, dim_n):
        return self.actor_weights[dim_n].dot(self.phi(x, dim_n))

    def get_actor_output(self, x):
        u = [self.get_actor_onedim(x, i) for i in range(self.dima)]
        return np.array(u)

    def phi(self, x, actor_idx):
        s1 = x[actor_idx]
        s2 = x[actor_idx + self.dima]
        return 2*np.tanh(0.5*np.array([s1, s2]))

    def save_iodata(self, path):
        dat = np.array(self.buffer)
        np.savez(path + '/qlearning.npz', t=np.array(self.timestamp), record=dat, actor=self.actor_logger, losses=self.loss_logger)


class Agent_DL:
    def __init__(self, dimx, dima):
        self.dimx = dimx
        self.dima = dima

        self.actor_weights = [np.array([-1, -2]) for i in range(dima)]

        self.states = tf.placeholder(tf.float32, [None, self.dimx])
        self.next_states = tf.placeholder(tf.float32, [None, self.dimx])
        self.us = tf.placeholder(tf.float32, [None, self.dima])
        self.rs = tf.placeholder(tf.float32, [None, 1])

        self.opt_actor_w1 = tf.Variable(tf.random_normal([dimx, 40], stddev=1, mean=0))
        self.opt_actor_w2 = tf.Variable(tf.random_normal([40, dima], stddev=0.01, mean=0))

        self._build_critic()
        optimizer = tf.train.AdamOptimizer(0.001)
        actor_loss = self.actor_loss1
        self.update_target_actor = optimizer.minimize(actor_loss, var_list=[self.opt_actor_w1, self.opt_actor_w2])
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.buffer = []
        self.timestamp = []
        self.actor_logger = []
        self.loss_logger = []

    def _build_critic(self):
        W1 = tf.Variable(tf.random_normal([self.dimx + self.dima, 50]))
        # b1 = tf.Variable(tf.zeros([20]))
        W2 = tf.Variable(tf.random_normal([50, 1]))

        ema = tf.train.ExponentialMovingAverage(0.99)
        self.update_target_critic = ema.apply([W1, W2, self.opt_actor_w1, self.opt_actor_w2])
        la1 = tf.tanh(tf.matmul(self.next_states, ema.average(self.opt_actor_w1)))
        actor_intermedia = tf.tanh(tf.matmul(la1, ema.average(self.opt_actor_w2)))
        # opt_u1 = actor_intermedia[:, 0] + actor_intermedia[:, 2]
        # opt_u2 = actor_intermedia[:, 1] + actor_intermedia[:, 3]
        target_critic_input = tf.concat([self.next_states, actor_intermedia], 1)
        critic_input = tf.concat([self.states, self.us], 1)

        layer = tf.tanh(tf.matmul(critic_input, W1))
        critic = tf.matmul(layer, W2)

        layer_t = tf.tanh(tf.matmul(target_critic_input, ema.average(W1)))
        target_critic = tf.matmul(layer_t, ema.average(W2))+ self.rs
        weight_decay = tf.add_n([tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)])

        self.critic_err = tf.reduce_mean(tf.square(target_critic - critic))
        self.critic_loss = self.critic_err  + 0.01*weight_decay

        la1_ = tf.tanh(tf.matmul(self.states, self.opt_actor_w1))
        actor_intermedia_ = tf.tanh(tf.matmul(la1_, self.opt_actor_w2))
        ci = tf.concat([self.states, actor_intermedia_], 1)
        layer = tf.tanh(tf.matmul(ci, W1))
        critic_ = tf.matmul(layer, W2)
        self.actor_loss1 = tf.reduce_mean(critic_)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.critic_update_op = optimizer.minimize(self.critic_loss, var_list=[W1, W2])

        self.verify_id_input = ci
        self.critic_predict = critic

    def _get_target_actor(self, state):
        wa1, wa2 = self.sess.run([self.opt_actor_w1, self.opt_actor_w2])
        # print wa
        m1 = np.tanh(np.matmul(state, wa1))
        m = np.tanh(np.matmul(m1, wa2))
        return m
        # return np.array([m[0]+m[2], m[1]+m[3]])

    def update_actor(self, s):
        ea = self.get_actor_output(s) - self._get_target_actor(s)
        self.actor_logger.append(self.actor_weights)
        # print ea
        for dimn in range(self.dima):
            phi = self.phi(s, dimn)
            self.actor_weights[dimn] = self.actor_weights[dimn] - 0.1*phi / ((1 + phi.dot(phi))**2) * ea[dimn] #0.32 

    def update_agent(self, batchsize):
        if len(self.buffer) > batchsize:
            batch = random.sample(self.buffer, batchsize)
        elif len(self.buffer) > 16:
            batch = self.buffer
        else:
            return
        train_inputs = np.array(batch)
        xs = train_inputs[:, :self.dimx]
        next_s = train_inputs[:, -self.dimx:]
        us = train_inputs[:, self.dimx:self.dimx + self.dima]
        rs = np.expand_dims(train_inputs[:, self.dimx + self.dima], 1)

        lc, la, _, _ = self.sess.run([self.critic_loss, self.actor_loss1, self.critic_update_op, self.update_target_actor], feed_dict={self.states: xs, self.next_states: next_s, self.us: us, self.rs: rs})
        print lc, la
        self.loss_logger.append(lc)
        # self.sess.run(self.update_target_actor, feed_dict={self.states: xs})
        self.sess.run(self.update_target_critic)

    def perceive(self, x, u, r, next_x, t):
        unit = np.hstack([x, u, r, next_x])

        self.timestamp.append(t)
        self.buffer.append(unit)

    def get_actor_onedim(self, x, dim_n):
        return self.actor_weights[dim_n].dot(self.phi(x, dim_n))

    def get_actor_output(self, x):
        u = [self.get_actor_onedim(x, i) for i in range(self.dima)]
        return np.array(u)

    def Q(self, s, a):
        state = np.array([s]) if len(s.shape) != 2 else s
        action = np.array([a]) if len(a.shape) != 2 else a
        Qsa = self.sess.run(self.critic_predict, feed_dict={self.states: state, self.us: action})
        return Qsa[0, 0]

    def phi(self, x, actor_idx):
        s1 = x[actor_idx]
        s2 = x[actor_idx + self.dima]
        return 2*np.tanh(0.5*np.array([s1, s2]))

    def save_iodata(self, path):
        dat = np.array(self.buffer)
        np.savez(path + '/qlearning.npz', t=np.array(self.timestamp), record=dat, actor=self.actor_logger, losses=self.loss_logger)
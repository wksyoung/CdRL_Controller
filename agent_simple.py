import tensorflow as tf
import numpy as np
import random

class Agent_Multidim:
    def __init__(self, dimx, dima, actor_num):
        self.dimx = dimx
        self.dima = dima

        actor_parmas = lambda ns, na, dev: [np.random.normal(loc=-dev, scale=0.6, size=(ns, )) for i in range(na)]
        #self.actor_weights = [actor_parmas(2, dima, 2) for i in range(actor_num)] # np.array([[-1], [-2]])
        self.actor_weights = [actor_parmas(2, dima, 1) for i in range(actor_num/4)] + [actor_parmas(2, dima, 2) for i in range(actor_num/4)] \
             + [actor_parmas(2, dima, 3) for i in range(actor_num/4)] + [actor_parmas(2, dima, 4) for i in range(actor_num/4)]

        self.states = tf.placeholder(tf.float32, [None, self.dimx])
        self.next_states = tf.placeholder(tf.float32, [None, self.dimx])
        self.us = tf.placeholder(tf.float32, [None, self.dima])
        self.rs = tf.placeholder(tf.float32, [None, 1])

        self.opt_actor_w1 = tf.Variable(tf.random_normal([dimx, 40], stddev=0.1, mean=0))
        self.opt_actor_w2 = tf.Variable(tf.random_normal([40, dima], stddev=0.01, mean=0))

        self._build_critic()
        self._build_identifier()
        optimizer = tf.train.AdamOptimizer(0.0005)
        actor_loss = self.actor_loss1 + self.actor_loss2
        self.update_target_actor = optimizer.minimize(actor_loss, var_list=[self.opt_actor_w1, self.opt_actor_w2])
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.buffer = []
        self.timestamp = []
        # self.actor_logger = []

    def _build_critic(self):
        la1 = tf.tanh(tf.matmul(self.next_states, self.opt_actor_w1))
        actor_intermedia = tf.tanh(tf.matmul(la1, self.opt_actor_w2))
        # opt_u1 = actor_intermedia[:, 0] + actor_intermedia[:, 2]
        # opt_u2 = actor_intermedia[:, 1] + actor_intermedia[:, 3]
        target_critic_input = tf.concat([self.next_states, actor_intermedia], 1)
        critic_input = tf.concat([self.states, self.us], 1)

        W1 = tf.Variable(tf.random_normal([self.dimx + self.dima, 50]))
        # b1 = tf.Variable(tf.zeros([20]))
        W2 = tf.Variable(tf.random_normal([50, 1]))

        layer = tf.tanh(tf.matmul(critic_input, W1))
        critic = tf.matmul(layer, W2)

        ema = tf.train.ExponentialMovingAverage(0.99)
        self.update_target_critic = ema.apply([W1, W2])

        layer_t = tf.tanh(tf.matmul(target_critic_input, ema.average(W1)))
        target_critic = tf.matmul(layer_t, ema.average(W2))+ self.rs
        weight_decay = tf.add_n([tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)])

        self.critic_loss = tf.reduce_mean(tf.square(target_critic - critic))  + 0.005*weight_decay
        self.actor_loss1 = tf.reduce_mean(target_critic)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.critic_update_op = optimizer.minimize(self.critic_loss, var_list=[W1, W2])

        self.verify_id_input = target_critic_input
        self.critic_predict = critic

    def _build_identifier(self):
        w1 = tf.Variable(tf.random_normal([self.dimx + self.dima, 20]))
        b1 = tf.Variable(tf.zeros([1, 20]))
        w2 = tf.Variable(tf.random_normal([20, self.dimx]))
        b2 = tf.Variable(tf.zeros([1, self.dimx]))

        id_input = tf.concat([self.states, self.us], 1)
        layer_id = tf.nn.sigmoid(tf.matmul(id_input, w1) + b1)
        self.predict_x = tf.matmul(layer_id, w2) + b2

        self.idloss = tf.reduce_mean(tf.reduce_sum(tf.square(self.next_states - self.predict_x), reduction_indices=[1]))
        self.train_id = tf.train.AdamOptimizer(0.01).minimize(self.idloss)

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
        # logger = [[],[]]
        for dimn in range(self.dima):
            self.actor_weights[idx][dimn] = self.actor_weights[idx][dimn] - 0.04*self.phi(s,dimn) * ea[dimn] #0.32
            for n in range(len(self.actor_weights)):
                #logger[dimn].append(self.actor_weights[n][dimn]) # adding in log
                if n != idx:
                    e = self.actor_weights[idx][dimn] - self.actor_weights[n][dimn]
                    self.actor_weights[n][dimn] = self.actor_weights[n][dimn] + 0.01*e #0.1
        # self.actor_logger.append(logger)

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

        lc, li, _, _ = self.sess.run([self.critic_loss, self.idloss, self.critic_update_op, self.train_id], feed_dict={self.states: xs, self.next_states: next_s, self.us: us, self.rs: rs})
        print lc, li
        self.sess.run(self.update_target_critic)
        self.sess.run(self.update_target_actor, feed_dict={self.next_states: next_s, self.rs: rs})

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
        np.savez(path + '/io_data.npz', t=np.array(self.timestamp), states=dat[:, :self.dimx], u=dat[:, self.dimx:self.dimx+self.dima], actor=self.actor_logger) 
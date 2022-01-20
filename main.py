#!/usr/bin/python

from numpy.core import machar
import rospy
from geometry_msgs.msg import Wrench
from hector_uav_msgs.msg import AttitudeErrors
from agent import Agent_Multidim, Agent_RL, Agent_DL
from ctm_utils import *
import threading
# import tensorflow as tf
import numpy as np
from record import Recorder
from test_machine import Test_Machine

PATH = '/home/pcl/wang/simulation_result/7-06'

class Classical_Machine:
    def __init__(self):
        self.critic_lock = threading.Lock()
        self.agent = Agent_DL(4, 2)
        self.learning_task = threading.Thread(target=self.learning_thread)
        self.States = {'state': None, 'action': None, 'step': 0}
        self.pub = rospy.Publisher('control_bf', Wrench, queue_size=5)
        rospy.Subscriber('states', AttitudeErrors, self.callback, queue_size= 5)
        self.start_time = rospy.Time.now()
        self.learning_task.start()
    
    def __del__(self):
        self.agent.save_iodata(PATH)

    def callback(self, data):
        doLearning = self.States['step'] < 2100
        next_state = np.array([-data.err_roll, -data.err_pitch, -data.dr, -data.dp])
        if (self.States['state'] is not None) and (self.States['action'] is not None) and doLearning:
            state = self.States['state']
            u = self.States['action']
            self.critic_lock.acquire()
            t = (rospy.Time.now() - self.start_time).to_sec()
            r = state.dot(state) * 0.5 + u.dot(u) * 0.01
            self.agent.perceive(state, u, r, next_state,t)
            self.agent.update_actor(state)
            self.critic_lock.release()
        state = next_state
        
        exp = np.random.normal(scale=0.5, size=(2,))
        if doLearning:
            u = self.agent.get_actor_output(state) + exp
        else:
            u = self.agent.get_actor_output(state)
            print self.agent.actor_weights[0], self.agent.actor_weights[1]
        wrench = Wrench()
        wrench.torque.x = u[0] * 20 + exp[0]
        wrench.torque.y = u[1] * 20 + exp[1]
        self.pub.publish(wrench)
        self.States['state'] = state
        self.States['action'] = u
        self.States['step'] += 1

    def learning_thread(self):
        self.train_step=0
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            # critic learning
            if self.States['step'] < 2100: #30
                self.critic_lock.acquire()
                self.agent.update_agent(128)
                self.critic_lock.release()
            self.train_step += 1
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                print('exiting learning thread...')

class Machine:
    def __init__(self, actor_n):
        self.critic_lock = threading.Lock()
        # self.sess = tf.Session()
        self.agent = Agent_Multidim(4, 2, actor_n) # Agent(2, 1, 10, n_actors, sess)

        # init = tf.compat.v1.global_variables_initializer()
        # self.sess.run(init)
        self.weights = np.ones([actor_n])
        self.ACTOR_N = actor_n
        self.States = {'state': None, 'action': None, 'step': 0}
        self.learning_task = threading.Thread(target=self.learning_thread)
        self.pub = rospy.Publisher('control_bf', Wrench, queue_size=5)
        rospy.Subscriber('states', AttitudeErrors, self.callback, queue_size= 5)
        self.start_time = rospy.Time.now()
        self.learning_task.start()

    def __del__(self):
        self.agent.save_iodata(PATH)

    def callback(self, data):
        # print 'callbackstep:', self.States['step']
        doLearning = self.States['step'] < 2100
        next_state = np.array([-data.err_roll, -data.err_pitch, -data.dr, -data.dp])
        if (self.States['state'] is not None) and (self.States['action'] is not None):
            state = self.States['state']
            u = self.States['action']
            self.critic_lock.acquire()
            t = (rospy.Time.now() - self.start_time).to_sec()
            r = state.dot(state) * 0.5 + u.dot(u) * 0.01
            self.agent.perceive(state, u, r, next_state,t)
            self.critic_lock.release()
        state = next_state
        # up-tree competition
        if self.States['step'] > 30: # give score when critic and identifier have acquired some knowledge
            self.critic_lock.acquire()
            scores = np.array([self.score(i, state) for i in range(self.ACTOR_N)])
            self.critic_lock.release()
            scores -= np.min(scores)
        else:
            scores = np.ones([self.ACTOR_N])
        if doLearning:
            winner_id = uptree(scores * self.weights)
            exp = np.random.normal(scale=0.6, size=(2,))
        else:
            winner_id = np.argmax(scores * self.weights)
            exp = np.zeros((2,))
            print self.agent.actor_weights[winner_id][0], self.agent.actor_weights[winner_id][1]
        # execute winner's control command to control the plant
        u = self.agent.get_actor_output(winner_id, state)
        wrench = Wrench()
        wrench.torque.x = u[0] * 20 + exp[0]
        wrench.torque.y = u[1] * 20 + exp[1]
        self.pub.publish(wrench)

        if doLearning:
            self.critic_lock.acquire()
            self.weights = sleeping_experts(state, u, self.weights, self.agent)
            self.critic_lock.release()

            self.critic_lock.acquire()
            self.agent.update_actor(winner_id, state)
            self.critic_lock.release()
        self.States['state'] = state
        self.States['action'] = u
        self.States['step'] += 1

    def score(self, n, s):
        a = self.agent.get_actor_output(n, s)
        return -self.agent.Q(s, a)

    def learning_thread(self):
        self.train_step=0
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            # critic learning
            if self.States['step'] < 2100: #30 self.States['step'] % 10 == 0 and 
                self.critic_lock.acquire()
                self.agent.update_agent(128)
                self.critic_lock.release()
            self.train_step += 1
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                print('exiting learning thread...')

if __name__ == '__main__':
    rospy.init_node('ai_controller', anonymous=True)
    rec = Recorder(30, PATH)
    machine = Machine(8)
    # machine = Classical_Machine()
    # machine.agent.save_iodata(PATH)
    # machine = Test_Machine(-2.55, -1.76, -2.43, -2.02)
    rospy.spin()
    machine.learning_task.join()

import numpy as np
import rospy
from geometry_msgs.msg import Wrench
from hector_uav_msgs.msg import AttitudeErrors

class Test_Machine:
    def __init__(self, kp_roll, kd_roll, kp_pitch, kd_pitch):
        self.States = {'state': None, 'action': None, 'step': 0}
        self.pub = rospy.Publisher('control_bf', Wrench, queue_size=5)
        rospy.Subscriber('states', AttitudeErrors, self.callback, queue_size= 5)
        self.start_time = rospy.Time.now()
        self.kp1 = kp_roll
        self.kd1 = kd_roll
        self.kp2 = kp_pitch
        self.kd2 = kp_pitch
    
    def __del__(self):
        self.agent.save_iodata(PATH)

    def callback(self, data):
        state = np.array([-data.err_roll, -data.err_pitch, -data.dr, -data.dp])
        
        K = np.array([[self.kp1,0,self.kd1,0], [0,self.kp2,0,self.kd2]])
        u = np.matmul(K, state)
        wrench = Wrench()
        wrench.torque.x = u[0] * 20
        wrench.torque.y = u[1] * 20
        self.pub.publish(wrench)
        self.States['state'] = state
        self.States['action'] = u
        self.States['step'] += 1

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf.transformations import *
import numpy as np

class Recorder:
    def __init__(self, time, savepath):
        rospy.Subscriber("ground_truth/state", Odometry, self.state_callback)
        self.positions = []
        self.velocities = []
        self.eulers = []
        self.timestamps = []
        self.done = False
        self.record_time = time
        self.path = savepath

    def state_callback(self, data):
        t = data.header.stamp.secs + data.header.stamp.nsecs / 1e9
        if t >= self.record_time:
            if not self.done:
                self.save_all(self.path + '/flight_record.npz')
                self.done = True
            print('record done!')
            return
        position = [data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z]
        vel = [data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z]
        euler = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        self.positions.append(position)
        self.velocities.append(vel)
        self.eulers.append(euler)
        self.timestamps.append(t)

    def save_all(self, filename):
        np.savez(filename, position=self.positions, velocity=self.velocities, time=self.timestamps, attitude=self.eulers)
        print('Record done!')
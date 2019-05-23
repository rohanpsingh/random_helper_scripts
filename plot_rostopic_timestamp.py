import rospy
import sys
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import scipy.misc
import numpy as np
from glob import glob
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
import re


    
def msg1Callback(msg):
    ts = msg.header.stamp.to_sec()
    time_stamp1.append(float(ts))
    print ("msg: \t", ts)
    print ("------")
    return

def msg2Callback(msg):
    ts = msg.header.stamp.to_sec()
    time_stamp2.append(float(ts))
    print ("msg: \t", ts)
    print ("------")
    return


if __name__ == '__main__':

    rospy.init_node('plot_timestamp', anonymous=False)

    time_stamp1 = []
    time_stamp2 = []    

    rospy.Subscriber(sys.argv[1], Imu, msg1Callback)
    rospy.Subscriber(sys.argv[2], Image, msg2Callback)
    

    rospy.spin()

    i = plt.figure(1)
    plt.plot(range(len(time_stamp1)), time_stamp1, 'r.')
    plt.plot(range(len(time_stamp2)), time_stamp2, 'g.')
    plt.xlabel("x-position")    
    plt.ylabel("y-position")
    plt.grid()
    plt.savefig("output.png", dpi=300)



    


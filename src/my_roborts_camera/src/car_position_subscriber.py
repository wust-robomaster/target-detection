#! /usr/bin/python2.7
# coding: utf-8

import rospy
from my_roborts_camera.msg import car_position


def publish_info_callback(msg):
    if(msg.color == 0):
        color = 'red'
    else:
        color = 'blue'
    rospy.loginfo("Subcribe car Info: color:%s  x:%f  y:%f",
                  color, msg.x, msg.y)


def msg_subscription():
    # ROS节点初始化
    rospy.init_node('Car_Str_Msg_Node', anonymous=True)

    # 创建一个Subscriber，订阅名为/car_position_info的topic，注册回调函数publish_info_callback
    rospy.Subscriber("/car_position_info", car_position, publish_info_callback)

    # 循环等待回调函数
    rospy.spin()


def main():
    msg_subscription()

main()

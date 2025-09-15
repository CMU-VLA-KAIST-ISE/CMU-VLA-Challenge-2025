#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

def main():
    rospy.init_node("user_query_publisher", anonymous=True)

    # latch=True: 구독자가 나중에 붙어도 마지막 메시지를 받아요.
    pub = rospy.Publisher("/user_query", String, queue_size=1, latch=True)
    msg = "Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door."
    #동시에 실행시켰을 때 parial scene graph보다 늦게 publish 하기 위해 10초 기다림.
    end = rospy.Time.now() + rospy.Duration(10.0)
    rate = rospy.Rate(20)
    while pub.get_num_connections() == 0 and rospy.Time.now() < end and not rospy.is_shutdown():
        rate.sleep()
    pub.publish(String(data=msg))
    rospy.loginfo("Published to /user_query: %s", msg)
    rospy.spin()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool

def main():
    rospy.init_node("exploration_done_dummy", anonymous=False)
    pub = rospy.Publisher("/exploration_done", Bool, queue_size=1, latch=True)
    delay = rospy.get_param("~delay_sec", 10.0)
    rospy.loginfo("Waiting %.1f sec, then publishing /exploration_done=True ...", delay)
    rospy.sleep(delay)
    pub.publish(Bool(data=True))
    rospy.loginfo("Published /exploration_done: True (latched)")
    rospy.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
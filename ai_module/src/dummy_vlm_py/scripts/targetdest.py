#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1차 목표로 먼저
/target_object_idx랑 scene graph받아서 
해당 object의 centroid 위치 좌표 출력하는 코드 작성
"""

import json
import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Point

#centroid 출력을 목표로 하는 class
#코드 실행 trigger는 target_obj_idx의 발행이 되어야함
class TargetCentroidPrinter:
    def __init__(self):
        self.last_sg = None
        self.last_idx = None 
        rospy.Subscriber("/partial_scene_graph_generator/partial_scene_graph",String,self.cb_sg,queue_size=1)
        rospy.Subscriber("/target_obj_idx",Int32,self.cb_idx,queue_size=10)
        self.pub_centroid = rospy.Publisher("/target_centroid",Point,queue_size=10,latch=True)
        rospy.loginfo("[CentroidPrinter] ready.")
    
    #scene_graph 호출시 -> 그냥 json 만 load
    def cb_sg(self, msg: String):
        try:
            self.last_sg = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("[CentroidPrinter] bad SG JSON: %s", e)

    #target_obj_idx가 call 되었을 떄-> trigger 가 되어서 centroid들고 와야함
    def cb_idx(self, msg: Int32):
        self.last_idx = int(msg.data)
        self.try_report()

    #centroid 좌표 추출 -> publish
    def try_report(self):
        if self.last_sg is None or self.last_idx is None:
            return
        centroid = self.lookup_centroid(self.last_sg, self.last_idx)
        if centroid is None:
            rospy.logwarn("[CentroidPrinter] object_id=%s not found or no centroid.", self.last_idx)
            return

        x, y, z = centroid
        rospy.loginfo("[CentroidPrinter] target_obj_id=%d centroid = [%.6f, %.6f, %.6f]",
                      self.last_idx, x, y, z)
        # publish Point for downstream nodes
        self.pub_centroid.publish(Point(x=x, y=y, z=z))

    #centroid 좌표 추출하는 코드
    @staticmethod
    def lookup_centroid(sg: dict, obj_id: int):
        resp=sg.get("response")
        object=resp.get(str(self.last_idx))
        centroid=object.get("centroid")
        return centroid

def main():
    rospy.init_node("target_centroid_printer", anonymous=False)
    TargetCentroidPrinter()
    rospy.spin()

if __name__ == "__main__":
    main()
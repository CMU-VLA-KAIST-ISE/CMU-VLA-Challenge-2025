#!/usr/bin/env python3
import rospy
import roslaunch
from std_msgs.msg import Int32,String,Bool
import os,queue
from pathlib import Path
from typing import Union, Sequence
import time

class LaunchOnEvent:
    def __init__(self):
        self._launch = None
        self._uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self._uuid)
        self.base=str(Path(__file__).resolve().parents[2])
        #메인스레드에서만 launch가 되는 오류가 있어서 일단 cb상에서는 queue에 쌓기만하고 main 에서 돌아가게 수정
        self.launch_queue=queue.Queue()
        self.question_type=rospy.Subscriber('/question_type', Int32, self._cb,queue_size=1)
        self.launch_type=rospy.Subscriber('/launch_type',Int32,self._cb,queue_size=1)
        #발행 시 현재 실행중인 launch 파일들 모두 shutdown
        self.shutdown_signal=rospy.Subscriber("/shutdown",Bool,self.cb_shutdown,queue_size=1)
        self.end_sig=rospy.Publisher("/end_signal",Bool,queue_size=1)
        self.is_end=False

    def _cb(self, msg : Int32):
        type=int(msg.data)
        rospy.loginfo(f"{msg.data}")
        try:
            self.launch_queue.put_nowait(type)
        except queue.Full:
            pass

    def run(self):
        while not rospy.is_shutdown():
            try:
                qt=self.launch_queue.get_nowait()
                self._handle(qt)
            except queue.Empty:
                pass

    def _handle(self, qt:int):
        if qt in(1,2):
            self.start_launch([f"{self.base}/color_scan_generation/launch/color_scan_generation.launch",
                              f"{self.base}/planning_node/launch/planning_node_1_2.launch",
                              f"{self.base}/gemini_api/launch/fusion_pipeline_SG_node_merge.launch",
                              f"{self.base}/dummy_vlm_py/launch/llm_prompt_builder.launch"]
                              )
            time.sleep(30)
            self.is_end=True
        elif qt==3:
            self.start_launch([f"{self.base}/color_scan_generation/launch/color_scan_generation.launch",
                                f"{self.base}/planning_node/launch/a_star_point_nav.launch",
                               f"{self.base}/gemini_api/launch/fusion_pipeline_SG_node_merge_task3.launch",
                               f"{self.base}/interaction_manager/launch/plan_task_3.launch"])
            self.is_end=True
        elif qt==0:
            self.start_launch([f"{self.base}/leo_vlm/launch/gpt_llm.launch"])
        elif qt==4:
            rospy.loginfo("!!!")
            self.start_launch([f"{self.base}/leo_vlm/launch/gpt_llm_4o.launch"])
    
    def start_launch(self, launch_files:Union[str, Sequence[str]]):
        self._launch=roslaunch.parent.ROSLaunchParent(self._uuid,launch_files)
        self._launch.start()
        rospy.loginfo("Launch started")
        
    def cb_shutdown(self,msg:Bool):
        if msg:    
            self.shutdown()
    
    def shutdown(self):
        if self._launch:
            try:
                self._launch.shutdown()
                rospy.loginfo("Launch shutdown.")
            except Exception as e:
                rospy.logerr(f"Failed to shutdown launch: {e}")
        if self.is_end:
            self.end_sig.publish(Bool(True))
            rospy.signal_shutdown("end_launch_pipeline")

if __name__ == "__main__":
    rospy.init_node("launch_on_event")
    node = LaunchOnEvent()
    rospy.on_shutdown(node.shutdown)
    node.run()
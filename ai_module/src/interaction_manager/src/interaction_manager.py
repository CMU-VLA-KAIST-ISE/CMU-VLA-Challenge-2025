#!/usr/bin/env python3
import rospy
import roslaunch
from std_msgs.msg import Int32,String,Bool
import os,queue
from pathlib import Path
from typing import Union, Sequence


class InteractionManager:
    def __init__(self):
        rospy.init_node('interaction_manager', anonymous=False)
        self.llm=rospy.Subscriber("/llm_response",String,self.cb_llm_response,queue_size=1)
        self.question_type_pub = rospy.Publisher('/question_type', Int32, queue_size=1,latch=True)
        self.question = rospy.Publisher("/challenge_question",String,queue_size=1,latch=True)
        self.gpt_launch=rospy.Publisher("/launch_type",Int32, queue_size=1)
        self.type_question=rospy.Publisher("/user_query",String,queue_size=1,latch=True)
        self.shutdown=rospy.Publisher("/shutdown",Bool,queue_size=1)
        self.published_qt=False
        rospy.loginfo("Interaction Manager initialized. Awaiting questions...")
    
    def run(self):
        print("Awaiting question...")
        while not rospy.is_shutdown():
            try:
                # 사용자 입력 받기
                user_input = input("> ")
                # 입력 정리 (surrogate 유니코드 문자만 제거)
                cleaned_input = user_input.encode('utf-8', errors='ignore').decode('utf-8').strip()
                self.question.publish(String(cleaned_input))
                self.gpt_launch.publish(Int32(data=0))
                if cleaned_input:
                    # 질문 타입 분류
                    prompt = f"""\
SYSTEM:
You are a strict single-digit classifier for household-scene questions. Your job is to classify the single input question into exactly one of the following categories:
1 = numerical — asks for a count(e.g., "How many…","Count the number of…").
2 = object_reference — identifies a specific object using relations/attributes/spatial terms(e.g., "Find the …").
3 = instruction_following — step-by-step navigation/action instructions (e.g., "Go…", "First…, then…", "Take the path…").
USER:
{cleaned_input}
TASK:
Read the user's single question and decide which single label (1/2/3) applies.

OUTPUT FORMAT (STRICT):
- Respond with EXACTLY ONE digit: 1, 2, or 3.
- Do not include any words, explanations, punctuation, or whitespace before/after the digit.
"""
                    self.type_question.publish(String(data=prompt))
            except KeyboardInterrupt:
                rospy.loginfo("Keyboard interrupt received, shutting down...")
                break
            except Exception as e:
                rospy.logerr(f"Error in interaction loop: {e}")
                print("Error occurred. Please try again.")
    
    def cb_llm_response(self,msg: String):
        self.type_question.unregister()
        if not self.published_qt:
            rospy.loginfo("[interaction_manager] llm response : %s",msg)
            type=int(msg.data)
            if type in (1,2):
                self.question_type_pub.publish(Int32(data=type))
                self.published_qt=True
                self.shutdown.publish(Bool(data=True))
            elif type==3:
                self.question_type_pub.publish(data=type)
                self.published_qt=True
            else:
                rospy.logerr(f"Unexpected question type : {type}")
if __name__ == "__main__":
    try:
        interaction_manager = InteractionManager()
        interaction_manager.run()
    except rospy.ROSInterruptException:
        pass
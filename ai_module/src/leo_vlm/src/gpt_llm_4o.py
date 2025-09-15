#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from openai import OpenAI
import base64

class GPT4oNode:
    def __init__(self):
        rospy.init_node("gpt_llm", anonymous=False)
        self.model = rospy.get_param("~model","gpt-4o")
        self.api_key_env = rospy.get_param("~api_key_env","OPENAI_API_KEY")
        self.system_instructions = rospy.get_param("~system_instructions", "")
        api_key = rospy.get_param("~openai_api_key", os.environ.get(self.api_key_env))
        if not api_key:
            rospy.logfatal(
                "OpenAI API key not found. Set env %s or ~openai_api_key param.",
                self.api_key_env,
            )
            raise SystemExit(1)
        self.client = OpenAI(api_key=api_key)
        # 퍼블리셔/서브스크라이버
        self.pub = rospy.Publisher("/llm_response", String, queue_size=10)
        self.sub = rospy.Subscriber("/user_query_4o", String, self.cb_query, queue_size=10)
        self.img = rospy.Subscriber("/user_image_4o",CompressedImage,self.cb_image,queue_size=10)
        self.use_image=False
        rospy.loginfo(
            "GPT4o node ready. Subscribing /user_query → publishing /llm_rersponse. "
            "Model=%s, KeyEnv=%s",
            self.model, self.api_key_env
        )

    def _build_prompt(self,text:str,with_image:bool):
        msgs = []
        if self.system_instructions:
            msgs.append({
                "role": "system",
                "content": [{"type": "input_text", "text": self.system_instructions}],
            })

        user_content = [{"type": "input_text", "text": text}]
        if with_image and self.last_image_b64:
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/{self.last_image_fmt};base64,{self.last_image_b64}",
            })
        msgs.append({"role": "user", "content": user_content})
        return msgs

    # 콜백에서 바로 API 호출
    def cb_query(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        try:
            payload = self._build_prompt(text, with_image=self.use_image)
            resp = self.client.responses.create(model=self.model, input=payload)
            out = getattr(resp, "output_text", None) or ""
            if not out:
                try:
                    out = resp.output[0].content[0].text
                except Exception:
                    out = ""
            if not out:
                out = "[LLM returned empty response]"
            self.pub.publish(String(out))
            rospy.loginfo("LLM OK (%d chars)", len(out))
        except Exception as e:
            err = f"[LLM error in gpt_llm_4o.py] {e}"
            rospy.logerr(err)
            self.pub.publish(String(err))
        finally:
            self.use_image = False


    def cb_image(self, msg):
        self.use_image=True
        self.last_image_fmt=(msg.format or "jpeg").lower()
        self.last_image_b64=base64.b64encode(msg.data).decode("utf-8")

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    GPT4oNode().spin()

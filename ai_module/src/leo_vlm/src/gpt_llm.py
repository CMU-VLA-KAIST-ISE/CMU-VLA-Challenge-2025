#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import rospy
import shutil
from std_msgs.msg import String
from sensor_msgs.msg import Image
from openai import OpenAI
import base64
from cv_bridge import CvBridge, CvBridgeError  # gpt: add cv_bridge to convert raw Image -> PNG bytes

class GPT5Node:
    def __init__(self):
        rospy.init_node("gpt_llm", anonymous=False)
        self.model = rospy.get_param("~model","gpt-5")
        self.api_key_env = rospy.get_param("~api_key_env","OPENAI_API_KEY")
        self.system_instructions = rospy.get_param("~system_instructions", "")
    # gpt: remove relativeness_dir dependency; we no longer read images from /tmp
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
        self.sub = rospy.Subscriber("/user_query", String, self.cb_query, queue_size=10)
        # self.img = rospy.Subscriber("/user_image", Image, self.cb_image, queue_size=10)  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        # gpt: now images are loaded from /tmp/rel automatically for non-classifier prompts
        self.rel_dir = rospy.get_param("~rel_dir", "/tmp/rel")
        self.bridge = CvBridge()  # keep for future use
        rospy.loginfo(
            "GPT5 node ready. Subscribing /user_query → publishing /llm_response. "
            "Model=%s, KeyEnv=%s",
            self.model, self.api_key_env
        )

    def _get_top_rel_images(self, topk=3):
        imgs = []
        try:
            if not os.path.isdir(self.rel_dir):
                rospy.loginfo(f"[GPT LLM] rel dir missing: {self.rel_dir}")
                return []
            cand = []
            for name in os.listdir(self.rel_dir):
                if not name.lower().endswith('.png'):
                    continue
                m = re.search(r"_(\d{1,3})\.png$", name)
                score = -1
                if m:
                    try:
                        score = int(m.group(1))
                    except Exception:
                        score = -1
                cand.append((score, os.path.join(self.rel_dir, name)))
            cand.sort(key=lambda x: (x[0] if isinstance(x[0], int) else -1), reverse=True)
            for _, p in cand[:topk]:
                try:
                    with open(p, 'rb') as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                    imgs.append(("png", b64))
                except Exception as e:
                    rospy.logwarn(f"[GPT LLM] fail read image {p}: {e}")
        except Exception as e:
            rospy.logwarn(f"[GPT LLM] scan rel dir failed: {e}")
        return imgs

    def _build_prompt(self,text:str,with_image:bool):
        msgs = []
        if self.system_instructions:
            msgs.append({
                "role": "system",
                "content": [{"type": "input_text", "text": self.system_instructions}],
            })
        #compress image여서 self.last_image_fmt는 보통 아마 png로 들어갈 것으로 생각
        user_content = [{"type": "input_text", "text": text}]
        # 분류용 단일 숫자 응답 프롬프트는 이미지 첨부 금지
        is_classifier = ("single-digit classifier" in text.lower()) or (re.search(r"EXACTLY ONE digit", text, flags=re.I) is not None)
        print('Is classifier prompt:', is_classifier)
        # gpt: attach up to 3 images from /tmp/rel for non-classifier prompts
        imgs = []
        if not is_classifier:
            imgs = self._get_top_rel_images(topk=3)
        rospy.loginfo(f"[GPT LLM] Include images: {len(imgs)} (classifier={is_classifier})")
        for fmt, b64 in imgs:
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/{fmt};base64,{b64}",
            })
        msgs.append({"role": "user", "content": user_content})
        for m in msgs:
            print(f"Message role={m['role']} content types: {[c['type'] for c in m['content']]}")
        return msgs

    # 콜백에서 바로 API 호출
    def cb_query(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        flag = 1
        while(flag):
            try:
                rospy.loginfo(f"[LLM] Prompt Text\n{text}")
                payload = self._build_prompt(text, with_image=False)
                resp = self.client.responses.create(model=self.model, input=payload)
                out = getattr(resp, "output_text", None) or ""
                if not out:
                    try:
                        out = resp.output[0].content[0].text
                        # print(out)
                    except Exception:
                        out = ""
                if not out:
                    out = "[LLM returned empty response]"
                self.pub.publish(String(out))
                rospy.loginfo("LLM OK (%d chars)", len(out))
                # rospy.loginfo("LLM Input: %s", text)
                # rospy.loginfo("LLM Output: %s", out)
                flag = 0
            except Exception as e:
                err = f"[LLM error in gpt_llm.py] {e}"
                rospy.logerr(err)
    # def cb_image(self, msg: Image):
    #     """원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]"""
    #     pass


    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    GPT5Node().spin()

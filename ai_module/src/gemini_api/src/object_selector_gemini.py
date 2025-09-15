#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from typing import Any, List, Optional

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from PIL import Image as PILImage
import google.generativeai as genai
import io

class ImageOnlyGeminiResponder:
    """
    Subscribes:
      ~image_topic        (sensor_msgs/Image)   [default: /partial_scene_graph_generator/current_image]
      ~instruction_topic  (std_msgs/String)     [default: /instruction]

    Publishes:
      ~answer_topic       (std_msgs/String)     [default: /gemini_answer]

    Params:
      ~model_name (str): Gemini model name (default: "gemini-2.5-flash")
      ~temperature (float), ~max_output_tokens (int)
      ~image_topic (str), ~instruction_topic (str), ~answer_topic (str)
      ~system_prompt (str): optional; prepend guidance for model behavior
    """

    def __init__(self):
        rospy.init_node("image_only_gemini_responder", anonymous=True)

        # -------- Parameters --------
        self.model_name = rospy.get_param("~model_name", "gemini-2.5-flash")
        self.temperature = float(rospy.get_param("~temperature", 0.1))
        self.max_output_tokens = int(rospy.get_param("~max_output_tokens", 1024))

        default_image_topic = "/partial_scene_graph_generator/current_image"
        default_instruction_topic = "/instruction"
        default_answer_topic = "/gemini_answer"

        image_topic = rospy.get_param("~image_topic", default_image_topic)
        instruction_topic = rospy.get_param("~instruction_topic", default_instruction_topic)
        answer_topic = rospy.get_param("~answer_topic", default_answer_topic)
        self.system_prompt = rospy.get_param(
            "~system_prompt",
            "You are a helpful vision-language assistant. "
            "Answer concisely based on the provided image and the instruction. "
            "If the image is missing, say you need an image."
        )

        # -------- Publishers / Subscribers --------
        self.pub_answer = rospy.Publisher(answer_topic, String, queue_size=10)
        self.sub_image = rospy.Subscriber(image_topic, RosImage, self._image_cb, queue_size=1)
        self.sub_instruction = rospy.Subscriber(instruction_topic, String, self._instruction_cb, queue_size=1)

        # -------- Buffers --------
        self.last_pil_image: Optional[PILImage.Image] = None
        self.last_instruction: Optional[str] = None

        # -------- Gemini setup --------
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            rospy.logwarn("[Gemini] GEMINI_API_KEY missing; requests will fail.")
        genai.configure(api_key=api_key)
        self.gemini = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "text/plain",
            },
        )

        rospy.loginfo(f"[Gemini] model={self.model_name}")
        rospy.loginfo(f"[Subscribe] image_topic={image_topic}")
        rospy.loginfo(f"[Subscribe] instruction_topic={instruction_topic}")
        rospy.loginfo(f"[Publish] answer_topic={answer_topic}")

    # ---------------- Image decoding without cv_bridge/opencv ----------------
    def _rosimg_to_pil(self, msg: RosImage) -> Optional[PILImage.Image]:
        """Convert sensor_msgs/Image to PIL.Image (supports rgb8/bgr8/rgba8/bgra8/mono8)."""
        try:
            w, h = int(msg.width), int(msg.height)
            enc = (msg.encoding or "").lower()
            step = int(msg.step)
            buf = bytes(msg.data) if isinstance(msg.data, (bytes, bytearray)) else bytes(bytearray(msg.data))

            if enc == "rgb8":
                mode, decoder, bpp = "RGB", None, 3
            elif enc == "bgr8":
                mode, decoder, bpp = "RGB", "BGR", 3
            elif enc == "rgba8":
                mode, decoder, bpp = "RGBA", None, 4
            elif enc == "bgra8":
                mode, decoder, bpp = "RGBA", "BGRA", 4
            elif enc in ("mono8", "8uc1"):
                mode, decoder, bpp = "L", None, 1
            else:
                rospy.loginfo(f"[image] unsupported encoding '{enc}', skipping image.")
                return None

            expected_step = w * bpp
            if step == expected_step:
                return (PILImage.frombytes(mode, (w, h), buf, "raw", decoder)
                        if decoder else PILImage.frombytes(mode, (w, h), buf))
            if step < expected_step:
                rospy.loginfo(f"[image] invalid step({step}) < expected({expected_step}), skipping.")
                return None

            # strip row padding
            out = bytearray(expected_step * h)
            for r in range(h):
                rs = r * step
                re = rs + expected_step
                rd = r * expected_step
                out[rd:rd + expected_step] = buf[rs:re]
            raw = bytes(out)
            return (PILImage.frombytes(mode, (w, h), raw, "raw", decoder)
                    if decoder else PILImage.frombytes(mode, (w, h), raw))
        except Exception as e:
            rospy.loginfo(f"[image] decode failed; proceeding without image: {e}")
            return None

    # ---------------- Callbacks ----------------
    def _image_cb(self, msg: RosImage):
        self.last_pil_image = self._rosimg_to_pil(msg)
        if self.last_pil_image is not None:
            rospy.loginfo(f"[image] received and decoded: {self.last_pil_image.size}")

    def _instruction_cb(self, msg: String):
        self.last_instruction = (msg.data or "").strip()
        if not self.last_instruction:
            rospy.loginfo("[instruction] empty; waiting for instruction...")
            return
        self._run_inference()

    # ---------------- Inference ----------------
    def _build_parts_text_then_pngbytes(self) -> list:
        """[전략1] 텍스트 먼저, 그 다음 PNG 바이너리 파트."""
        prompt_text = (
            f"{self.system_prompt}\n\n"
            "Answer concisely in plain text.\n"
            f"INSTRUCTION:\n{self.last_instruction or '(none)'}"
        )
        parts = [prompt_text]

        # 이미지가 있으면 PNG 바이트로 첨부 (너비 1024 이내로 다운스케일)
        if self.last_pil_image is not None:
            img = self.last_pil_image
            try:
                if img.width > 1024:
                    h = int(img.height * (1024.0 / img.width))
                    img = img.resize((1024, h))
            except Exception:
                pass
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            parts.append({"mime_type": "image/png", "data": buf.getvalue()})
        return parts


    def _build_parts_text_then_pil(self) -> list:
        """[전략2] 텍스트 먼저, 그 다음 PIL 이미지 객체 그대로."""
        prompt_text = (
            f"{self.system_prompt}\n\n"
            "Answer concisely in plain text.\n"
            f"INSTRUCTION:\n{self.last_instruction or '(none)'}"
        )
        parts = [prompt_text]
        if self.last_pil_image is not None:
            img = self.last_pil_image
            try:
                if img.width > 1024:
                    h = int(img.height * (1024.0 / img.width))
                    img = img.resize((1024, h))
            except Exception:
                pass
            parts.append(img)
        return parts


    def _run_inference(self):
        if self.last_pil_image is None:
            self.pub_answer.publish(String(data="No image available yet. Please send an image."))
            return

        strategies = [
            ("text+pngbytes", self._build_parts_text_then_pngbytes),
            ("text+pil", self._build_parts_text_then_pil),
            ("text-only", lambda: [f"{self.system_prompt}\n\n"
                                "Answer concisely in plain text.\n"
                                f"INSTRUCTION:\n{self.last_instruction or '(none)'}"])
        ]

        # 호출 공통 옵션(여유 있게)
        gen_cfg = {
            "temperature": self.temperature,
            "max_output_tokens": max(self.max_output_tokens, 1024),
            "response_mime_type": "text/plain",
        }

        for name, builder in strategies:
            try:
                parts = builder()
                # 모델 객체는 __init__에서 만든 self.gemini(=GenerativeModel)
                # generation_config를 매 호출마다 override하려면 아래처럼:
                resp = self.gemini.generate_content(parts, generation_config=gen_cfg, stream=False)

                # 메타 로깅
                fr = None
                if getattr(resp, "candidates", None):
                    fr = getattr(resp.candidates[0], "finish_reason", None)
                mv = getattr(resp, "model_version", None)
                rospy.loginfo(f"[gemini] strategy={name}, finish_reason={fr}, model_version={mv}")

                text = self._extract_text(resp).strip()
                if text and not text.startswith("GenerateContentResponse("):
                    self.pub_answer.publish(String(data=text))
                    rospy.loginfo(f"[publish] answer({name}): {text[:200]}{'...' if len(text)>200 else ''}")
                    return  # 성공 시 종료
                else:
                    rospy.loginfo(f"[gemini] empty text with strategy={name}; trying next...")
            except Exception as e:
                rospy.loginfo(f"[gemini] strategy={name} failed: {e}")
                # 다음 전략으로 넘어감

        # 모든 전략이 실패했다면
        self.pub_answer.publish(String(
            data="(no textual answer from model; tried text+pngbytes, text+pil, text-only)"
        ))


    def _build_gemini_prompt(self) -> List[Any]:
        instruction = self.last_pil_image and (self.last_instruction or "(none)") or "(no image)"
        prompt_text = (
            f"{self.system_prompt}\n\n"
            "Answer concisely in plain text.\n"
            f"INSTRUCTION:\n{self.last_instruction or '(none)'}"
        )

        # 이미지 → 텍스트 순서
        parts: List[Any] = []
        # 이미지 파트를 명시적 바이너리로
        img_buf = io.BytesIO()
        self.last_pil_image.save(img_buf, format="PNG")
        parts.append({"mime_type": "image/png", "data": img_buf.getvalue()})
        # 텍스트 파트
        parts.append(prompt_text)
        return parts


    # ---------------- Utilities ----------------
    def _extract_text(self, response: Any) -> str:
        """
        Gemini 응답에서 사람이 읽을 수 있는 텍스트를 최대한 뽑아낸다.
        - parts[].text 우선
        - candidates[].grounding(없을 때도 많음) 무시
        - 없으면 finish_reason 등 메타로 요약 메시지 생성
        """
        try:
            # 1) candidates → parts → text
            if getattr(response, "candidates", None):
                cand0 = response.candidates[0]
                content = getattr(cand0, "content", None)
                if content and getattr(content, "parts", None):
                    texts = []
                    for p in content.parts:
                        # part는 text 혹은 inline_data 등을 가질 수 있음
                        t = getattr(p, "text", None)
                        if t:
                            texts.append(t)
                    if texts:
                        return "\n".join(texts).strip()

                # 2) parts가 없거나 text가 비었을 때: finish_reason 등으로 힌트 메시지
                fr = getattr(cand0, "finish_reason", None) or "UNKNOWN"
                return f"(no text in candidates; finish_reason={fr})"

            # 3) 구 SDK에서 지원하는 shortcut
            t = getattr(response, "text", None)
            if t:
                return t.strip()

            # 4) 아무것도 없으면 요약
            mv = getattr(response, "model_version", None)
            return f"(empty response; model_version={mv or 'unknown'})"

        except Exception as e:
            # 5) 완전 예외면 예외 메시지 반환 (객체 덤프 금지)
            return f"(extract_text error: {e})"


if __name__ == "__main__":
    try:
        node = ImageOnlyGeminiResponder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

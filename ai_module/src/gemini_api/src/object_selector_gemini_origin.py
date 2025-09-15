#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import rospy
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image as RosImage
# from geometry_msgs.msg import PoseStamped  # 필요하면 사용

from PIL import Image as PILImage
import google.generativeai as genai


class ObjectSelectorGemini:
    """
    Subscribes (defaults):
      /partial_sg_generator/current_image        (sensor_msgs/Image)
      /partial_sg_generator/partial_scene_graph  (std_msgs/String, JSON)

      ※ 위 두 토픽은 파라미터로 변경 가능:
         ~image_topic (str), ~sg_topic (str)

    Publishes:
      /obj_id (std_msgs/Int32)

    Params:
      ~inference_mode (int): 1=image+SG, 2=SG only
      ~model_name (str): Gemini model name
      ~temperature (float), ~max_output_tokens (int)
      ~image_topic (str), ~sg_topic (str), ~instruction_topic (str)
    """

    REL_WORDS = ["near", "closest", "above", "below", "farthest", "nearest", "nearby"]

    def __init__(self):
        rospy.init_node("object_selector_gemini", anonymous=True)

        # Parameters
        self.inference_mode = rospy.get_param("~inference_mode", 1)
        self.model_name = rospy.get_param("~model_name", "gemini-2.5-flash")
        self.temperature = float(rospy.get_param("~temperature", 0.1))
        self.max_output_tokens = int(rospy.get_param("~max_output_tokens", 128))

        # Topic parameters (defaults match publisher with private "~" under node name "partial_sg_generator")
        default_image_topic = "/partial_sg_generator/current_image"
        default_sg_topic = "/partial_sg_generator/partial_scene_graph"
        default_instruction_topic = "/instruction"

        image_topic = rospy.get_param("~image_topic", default_image_topic)
        sg_topic = rospy.get_param("~sg_topic", default_sg_topic)
        instruction_topic = rospy.get_param("~instruction_topic", default_instruction_topic)

        # Publishers / Subscribers
        self.pub_obj_id = rospy.Publisher("/obj_id", Int32, queue_size=10)
        self.sub_image = rospy.Subscriber(image_topic, RosImage, self._image_cb, queue_size=1)
        self.sub_sg = rospy.Subscriber(sg_topic, String, self._sg_cb, queue_size=1)
        self.sub_instruction = rospy.Subscriber(instruction_topic, String, self._instruction_cb, queue_size=1)
        # self.sub_pose = rospy.Subscriber("/partial_sg_generator/current_pose", PoseStamped, self._pose_cb, queue_size=1)  # 필요시

        # Buffers
        self.last_pil_image: Optional[PILImage.Image] = None
        self.last_instruction: Optional[str] = None
        self.sg_objects: List[Dict[str, Any]] = []
        self.sg_rels: List[Tuple[int, str, int]] = []

        # Gemini setup
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

        rospy.loginfo(f"[Gemini] model={self.model_name}, inference_mode={self.inference_mode}")
        rospy.loginfo(f"[Subscribe] image_topic={image_topic}")
        rospy.loginfo(f"[Subscribe] sg_topic={sg_topic}")
        rospy.loginfo(f"[Subscribe] instruction_topic={instruction_topic}")

    # ---------------- Image decoding without cv_bridge/opencv ----------------
    def _rosimg_to_pil(self, msg: RosImage) -> Optional[PILImage.Image]:
        """Convert sensor_msgs/Image to PIL.Image using raw bytes only (supports rgb8/bgr8/rgba8/bgra8/mono8)."""
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

    def _sg_cb(self, msg: String):
        ok = self._parse_partial_sg(msg.data)
        if not ok:
            rospy.loginfo("[partial_sg] invalid or empty; proceeding without SG.")

    # def _pose_cb(self, msg: PoseStamped):
    #     pass  # 필요하면 활용

    def _instruction_cb(self, msg: String):
        self.last_instruction = (msg.data or "").strip()
        if not self.last_instruction:
            rospy.loginfo("[instruction] empty; waiting for instruction...")
            return
        self._run_inference()

    # ---------------- SG parsing ----------------
    def _parse_partial_sg(self, sg_raw: str) -> bool:
        try:
            data = json.loads(sg_raw)
        except Exception:
            return False

        self.sg_objects = []
        objs = data.get("objects", [])
        if isinstance(objs, list):
            for o in objs:
                if not isinstance(o, dict):
                    continue
                oid = o.get("object_id")
                try:
                    oid_int = int(oid)
                except Exception:
                    continue
                labels = []
                for k in ["raw_label", "nyu_label", "nyu40_label"]:
                    v = o.get(k)
                    if isinstance(v, str) and v.strip():
                        labels.append(v.strip().lower())
                color_labels = []
                if isinstance(o.get("color_labels"), list):
                    for c in o["color_labels"]:
                        if isinstance(c, str) and c != "N/A":
                            color_labels.append(c.strip().lower())
                self.sg_objects.append({
                    "object_id": oid_int,
                    "labels": list(dict.fromkeys(labels)),
                    "color_labels": color_labels,
                    "raw": o,
                })

        self.sg_rels = []
        rels = data.get("relationships", [])
        if isinstance(rels, list):
            for r in rels:
                if not (isinstance(r, list) and len(r) == 3):
                    continue
                s, rel, d = r
                try:
                    s_i, d_i = int(s), int(d)
                except Exception:
                    continue
                if not isinstance(rel, str):
                    continue
                self.sg_rels.append((s_i, rel.lower().strip(), d_i))

        return len(self.sg_objects) > 0

    # ---------------- Inference ----------------
    def _run_inference(self):
        if not self.last_instruction:
            rospy.loginfo("[inference] no instruction; skipping.")
            return

        if self.inference_mode == 1:
            if self.last_pil_image is None:
                rospy.loginfo("[inference] image not available; proceeding without it.")
            if not self.sg_objects:
                rospy.loginfo("[inference] scene graph not available; Gemini may still output an id.")
        else:
            if not self.sg_objects:
                rospy.loginfo("[inference] scene graph not available; cannot validate/publish id.")

        try:
            parts = self._build_gemini_prompt()
            response = self.gemini.generate_content(parts, stream=False)
            text = self._extract_text(response).strip()
            oid = self._extract_first_int(text)

            if oid is None:
                rospy.loginfo(f"[Gemini] failed to parse integer from response: '{text}'")
                return
            if self.sg_objects and not self._is_oid_in_sg(oid):
                rospy.loginfo(f"[Gemini] chosen id {oid} not in scene graph; ignoring.")
                return

            self.pub_obj_id.publish(Int32(data=int(oid)))
            rospy.loginfo(f"[publish] obj_id={int(oid)}")
        except Exception as e:
            rospy.loginfo(f"[Gemini] inference failed: {e}")

    def _will_send_image(self) -> bool:
        return (self.inference_mode == 1) and (self.last_pil_image is not None)

    # ---------------- Prompt building ----------------
    def _build_gemini_prompt(self) -> List[Any]:
        sg_snippet = self._format_sg_for_prompt(max_objects=200, max_rels=400)

        rules = [
            "Output ONLY one integer: the selected object_id.",
            "Choose an id that exists in SCENE_GRAPH.objects[].object_id.",
            "Match INSTRUCTION to object labels (raw_label, nyu_label, nyu40_label) and color_labels.",
            "If relation words appear (near, closest, above, below, farthest, nearest, nearby), use SCENE_GRAPH.relationships to disambiguate.",
            "If still ambiguous, pick the smallest object_id among top candidates.",
            "No explanation. No extra characters. Only the integer.",
        ]
        if self.inference_mode == 2:
            rules.append("Do NOT use the image. Use ONLY the scene graph.")

        prompt_text = (
            "You are an object selector for a robot.\n\n"
            "INSTRUCTION (free text):\n"
            f"{self.last_instruction or '(none)'}\n\n"
            "SCENE_GRAPH (partial; schema-aligned):\n"
            f"{sg_snippet}\n\n"
            "DECISION RULES:\n- " + "\n- ".join(rules) + "\n\n"
            "OUTPUT:\n"
            "Only the integer object_id."
        )

        parts: List[Any] = [prompt_text]
        if self._will_send_image():
            parts.append(self.last_pil_image)
        return parts

    def _format_sg_for_prompt(self, max_objects: int = 200, max_rels: int = 400) -> str:
        objs_out = []
        for o in self.sg_objects[:max_objects]:
            objs_out.append({
                "object_id": o["object_id"],
                "raw_label": o["raw"].get("raw_label"),
                "nyu_label": o["raw"].get("nyu_label"),
                "nyu40_label": o["raw"].get("nyu40_label"),
                "color_labels": o.get("color_labels", []),
            })
        rels_out = [[s, r, d] for (s, r, d) in self.sg_rels[:max_rels]]
        try:
            return json.dumps({"objects": objs_out, "relationships": rels_out}, ensure_ascii=False, indent=2)
        except Exception:
            return str({"objects": objs_out, "relationships": rels_out})

    # ---------------- Utilities ----------------
    def _extract_text(self, response: Any) -> str:
        try:
            candidates = getattr(response, "candidates", None)
            if candidates:
                cand0 = candidates[0]
                content = getattr(cand0, "content", None)
                if content and getattr(content, "parts", None):
                    texts = []
                    for p in content.parts:
                        t = getattr(p, "text", None)
                        if t:
                            texts.append(t)
                    txt = "\n".join(texts).strip()
                    if txt:
                        return txt
            t = getattr(response, "text", "") or ""
            return t
        except Exception:
            return str(response)

    def _extract_first_int(self, txt: str) -> Optional[int]:
        m = re.search(r"(-?\d+)", txt)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _is_oid_in_sg(self, oid: int) -> bool:
        return any(o.get("object_id") == oid for o in self.sg_objects)


if __name__ == "__main__":
    try:
        node = ObjectSelectorGemini()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
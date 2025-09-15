#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json
import base64  # gpt: for optional direct base64 if needed in future
from collections import deque  # gpt: use deque to cap images
from std_msgs.msg import String, Int32, Bool
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Quaternion
import tf.transformations as tft
from cv_bridge import CvBridge  # gpt: convert Image -> numpy/bytes when re-publishing
import math
import os
import shutil


class LLMPromptBuilder:
    def __init__(self):
        rospy.init_node("llm_prompt_builder", anonymous=False)

        # Params
        self.edge_ths = int(rospy.get_param("~edge_ths", 10))

        # State
        self.last_partial_sg = None
        self.last_question = ""
        self.last_qtype = None
        self.qtype1_step = 1  # qtype 1의 단계 추적 (1: 인덱스 찾기, 2: 개수 세기)

        # Subscribers
        rospy.Subscriber("/fusion/scene_graph_merged", String, self.on_partial_sg, queue_size=1)
        rospy.Subscriber("/fusion/scene_graph_rich_merged", String, self.on_partial_rich_sg, queue_size=1)
        rospy.Subscriber("/challenge_question", String, self.on_question, queue_size=1)
        rospy.Subscriber("/question_type", Int32, self.on_qtype, queue_size=1)
        rospy.Subscriber("/exploration_done", Bool, self.on_exploration_done, queue_size=1)
        rospy.Subscriber("/llm_response", String, self.on_llm_response, queue_size=1)

    # gpt: subscribe for images and rel_score
    # rospy.Subscriber("/image_scenegraph", Image, self.on_image, queue_size=5)  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
    # rospy.Subscriber("/rel_score", Int32, self.on_rel_score, queue_size=5)      # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        # gpt: keep only top-K images by rel_score
        self.K = int(rospy.get_param("~topk_images", 3))  # gpt: configurable, default 3
        self.image_buf = deque(maxlen=50)  # gpt: store (stamp, Image)
        self.score_buf = deque(maxlen=50)  # gpt: store (stamp, score)
        self.top_images = []               # gpt: cached top-k pairs [(score, Image)]
        self.bridge = CvBridge()           # gpt: cv_bridge instance

        # Publisher
        self.pub_prompt = rospy.Publisher("/user_query", String, queue_size=10, latch=True)
    # self.pub_user_image = rospy.Publisher("/user_image", Image, queue_size=5)  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        self.launch_gpt = rospy.Publisher("/launch_type", Int32, queue_size=1)
        self.numerical_output = rospy.Publisher("/numerical_response", Int32, queue_size=1, latch=True)
        self.object_reference_output = rospy.Publisher("/selected_object_marker", Marker, queue_size=1, latch=True)
        self.shutdown_sig = rospy.Publisher("/shutdown", Bool, queue_size=1)

        rospy.loginfo("[LLM Prompt Builder] ready.")
    
    def on_image(self, msg:Image):
        # gpt: buffer images with header.stamp to pair with scores
        self.image_buf.append((msg.header.stamp.to_sec() if msg.header else rospy.Time.now().to_sec(), msg))
        # self._refresh_top_k()
        rospy.loginfo(f"[LLM Prompt Builder] Received image. buf={len(self.image_buf)} top={len(self.top_images)}")
        
    def on_rel_score(self, msg:Int32):
        # gpt: buffer score with current time as coarse stamp
        # self.score_buf.append((rospy.Time.now().to_sec(), int(msg.data)))  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        # self._refresh_top_k()                                              # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        # rospy.loginfo(f"[LLM Prompt Builder] Received rel score: {msg.data}. scores={len(self.score_buf)} top={len(self.top_images)}")  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        pass  # gpt: rel_score는 fusion에서 API로만 사용하고 topic은 사용하지 않음

    # def _refresh_top_k(self):
    #     """gpt: Match latest scores to nearest images by time and keep top-K images by score."""
    #     if not self.image_buf or not self.score_buf:
    #         print("[LLM Prompt Builder] No images or scores to match.")
    #         return
    #     pairs = [] 
        
    #     for images, scores in zip(self.image_buf, self.score_buf):
    #         t_img, img = images
    #         t_sc, sc = scores
    #         print(f'[LLM Prompt Builder] loaded {t_img}, {t_sc}, {sc}')
    #         dt = abs(t_img - t_sc)
    #         pairs.append((sc, img))
    #     pairs.sort(key= lambda x: x[0], reverse=True) # sort by score
    #     self.top_images = pairs[: self.K]
        # # simple nearest neighbor matching (most recent pairs)
        # images = list(self.image_buf)
        # scores = list(self.score_buf)
        # pairs = []
        # for t_img, img in images:
        #     print(f'[LLM Prompt Builder] loaded {t_img}')
        #     # find nearest score in time
        #     best = None
        #     best_dt = 1e9
        #     for t_sc, sc in scores:
        #         print(f'[LLM Prompt Builder] {t_sc}, {sc}')
        #         dt = abs(t_img - t_sc)
        #         if dt < best_dt:
        #             best_dt = dt
        #             best = sc
        #     if best is not None:
        #         pairs.append((best, img))
        # # sort by score desc, take top-K
        # pairs.sort(key=lambda x: x[0], reverse=True)
        # self.top_images = pairs[: self.K]

    # ------------------- Subscribers Callbacks -------------------
    def on_partial_sg(self, msg: String):
        """Scene Graph를 받아 전처리 후 저장"""
        try:
            partial_sg_json = json.loads(msg.data)
            self.last_partial_sg_json = partial_sg_json
            self.last_partial_sg = self.preprocess_sg(partial_sg_json)
        except Exception as e:
            rospy.logwarn(f"[LLM Prompt Builder] Failed to parse or preprocess scene graph: {e}")

    def on_partial_rich_sg(self, msg: String):
        """Scene Graph를 받아 전처리 후 저장"""
        try:
            partial_rich_sg_json = json.loads(msg.data)
            self.last_partial_rich_sg_json = partial_rich_sg_json
            self.last_partial_rich_sg = self.preprocess_rich_sg(partial_rich_sg_json)
            # self.pub_done.publish(Bool(data=True))
        except Exception as e:
            rospy.logwarn(f"[LLM Prompt Builder] Failed to parse or preprocess scene graph: {e}")

    def on_question(self, msg: String):
        self.last_question = (msg.data or "").strip()

    def on_qtype(self, msg: Int32):
        self.last_qtype = int(msg.data)

    def on_llm_response(self, msg: String):
        """
        QType 1의 1단계 응답(객체 인덱스)을 받아 2단계 프롬프트를 생성
        """
        rospy.loginfo("[llm_prompt_builder] llm response : %s", msg)
        # gpt: simplify — we no longer do multi-step counting here
        # If any response arrives while waiting, just forward shutdown to close GPT-only launch.

        if self.qtype1_step==3:
            self.numerical_output.publish(Int32(data=msg.data))
            try:
                rel_dir = rospy.get_param("~rel_dir", "/tmp/rel")
                if os.path.isdir(rel_dir):
                    shutil.rmtree(rel_dir)
                os.makedirs(rel_dir, exist_ok=True)
                rospy.loginfo(f"[llm_prompt_builder] reset rel dir: {rel_dir}")
            except Exception as e:
                rospy.logwarn(f"[llm_prompt_builder] failed to reset rel dir: {e}")
            self.shutdown_sig.publish(Bool(data=True))
        elif self.last_qtype==2:
            #Marker 양식에 맞도록 post processing
            text=str(msg.data).strip()

            rich_not = True
            if rich_not:
                item = next((d for d in self.last_partial_sg_json['objects'] if d['id'] == int(text)), None)
                center = (item['center'][0], item['center'][1], item['center'][2])
                size = (item['aabb']['max'][0] - item['aabb']['min'][0], item['aabb']['max'][1] - item['aabb']['min'][1], item['aabb']['max'][2] - item['aabb']['min'][2])
                yaw = 0.
            else:
                item = next((d for d in self.last_partial_rich_sg_json if d['object_id'] == int(text)), None)
                center, size, yaw = self.obb_from_corners_xyyaw_no_numpy(item ['bbox'])

            rospy.loginfo(f"[LLM Prompt Builder] ANSWER: {center, size}")
            # ok,center,size=self._parse_aabb_and_center_size(text)
            # if not ok:
            #     rospy.logerr("Failed to parse 8 bbox points from LLM output.")
            #     return
            marker = self._make_cube_marker(center=center, size=size, yaw_rad=yaw,rgba=(1.0, 0.0, 0.0, 0.4), marker_id=0)
            print(marker)
            self.object_reference_output.publish(marker)
            try:
                rel_dir = rospy.get_param("~rel_dir", "/tmp/rel")
                if os.path.isdir(rel_dir):
                    shutil.rmtree(rel_dir)
                os.makedirs(rel_dir, exist_ok=True)
                rospy.loginfo(f"[llm_prompt_builder] reset rel dir: {rel_dir}")
            except Exception as e:
                rospy.logwarn(f"[llm_prompt_builder] failed to reset rel dir: {e}")
            self.shutdown_sig.publish(Bool(data=True))
        else:
            rospy.loginfo("[LLM Prompt Builder] NotImplementedError On LLM Response")
            self.shutdown_sig.publish(Bool(data=True))

    def on_exploration_done(self, msg: Bool):
        """탐색 완료 신호를 받으면, 저장된 정보를 바탕으로 첫 프롬프트를 생성"""
        if not msg.data:
            return
        rospy.loginfo("[LLM Prompt Builder] Exploration done. Building initial prompt.")
        if not self.last_question:
            rospy.logwarn("[LLM Prompt Builder] Missing question; won't trigger GPT.")
            return
        # gpt: launch GPT node now
        self.launch_gpt.publish(Int32(data=0))
    # gpt: publish top-K images first, then the final single-step question as user_query
    # sent = 0
    # for sc, img in self.top_images:
    #     try:
    #         # forward raw Image as-is to keep quality
    #         self.pub_user_image.publish(img)
    #         sent += 1
    #     except Exception as e:
    #         rospy.logwarn(f"[LLM Prompt Builder] failed to publish user_image: {e}")
    # rospy.loginfo(f"[LLM Prompt Builder] Published {sent} user_image(s) to GPT.")
    # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        # Build a concise prompt that uses the scene graph context
        # prompt = self.build_scene_prompt(
        #     question=self.last_question,
        #     partial_sg=self.last_partial_sg or {"objects":[], "relationships":[], "scene_name":"Unknown"},
        #     guidance="Answer the user's request as best as possible using the scene graph and the provided images. Pay more focus on images, rather than the scene graph.",
        #     output_format=(
        #         "OUTPUT: Provide a concise answer."
        #     )
        # )
        if self.last_qtype == 1:
            self.qtype1_step = 1 # 1단계 시작
            prompt = self.build_scene_prompt(
                question=self.last_question,
                partial_sg=self.last_partial_sg,
                partial_rich_sg=self.last_partial_rich_sg,
                guidance="Output the count as a single integer.",
                output_format=(
                    "OUTPUT FORMAT (STRICT):\n"
                    "- Respond ONLY with a single integer. No words, no explanations."
                )
            )
            # 다음 단계를 위해 상태 변경
            self.qtype1_step = 3
        elif self.last_qtype == 2:
            prompt = self.build_scene_prompt(
                question=self.last_question,
                partial_sg=self.last_partial_sg,
                partial_rich_sg=self.last_partial_rich_sg,
                guidance="Output the only one object id as a single integer.",
                output_format=(
                    "OUTPUT FORMAT (STRICT):\n"
                    "- Respond ONLY with a single integer. No words, no explanations."
                )
            )
        else: # qtype 3 등 기타
            prompt = "This question type is not implemented yet."
        if prompt:
            self.pub_prompt.publish(String(data=prompt))
            rospy.loginfo(f"[LLM Prompt Builder] Published initial prompt for QType {self.last_qtype}.")

    # ------------------- Helper Methods -------------------

    def obb_from_corners_xyyaw_no_numpy(self, corners):
        """
        corners: 길이 8의 리스트, 각 원소는 [x,y,z] (OBB의 8개 꼭짓점)
                가정: 바닥면은 xy 평면에 평행하며 회전은 z축(yaw)만 존재
        반환:
          center: (cx, cy, cz)
          lengths: (lx, ly, lz)   # 로컬 x, y, z 순서
          yaw: float (radians)    # +x -> +y 반시계
        """
        if len(corners) != 8:
            raise ValueError("corners must have length 8")

        # --- center ---
        sx = sy = sz = 0.0
        for x, y, z in corners:
            sx += x;
            sy += y;
            sz += z
        center = (sx / 8.0, sy / 8.0, sz / 8.0)

        # --- lz ---
        zs = [p[2] for p in corners]
        lz = max(zs) - min(zs)

        # --- xy의 4개 고유 코너 추출 ---
        # (입력은 위/아래 면이 같은 xy를 공유하므로 4개만 필요)
        uniq_xy = []
        seen = set()
        for x, y, _ in corners:
            key = (float(x), float(y))
            if key not in seen:
                seen.add(key)
                uniq_xy.append((x, y))
        if len(uniq_xy) != 4:
            # 부동소수 오차가 크다면 반올림 키를 쓰자
            seen.clear()
            uniq_xy = []
            for x, y, _ in corners:
                key = (round(x, 9), round(y, 9))
                if key not in seen:
                    seen.add(key)
                    uniq_xy.append((x, y))
        if len(uniq_xy) != 4:
            raise ValueError("Could not get 4 unique XY corners")

        # 한 코너 p0에서 가장 가까운 두 코너를 찾으면 그 두 변이 직교 변(인접 변)
        p0 = uniq_xy[0]
        dists = []
        for i in range(1, 4):
            xi, yi = uniq_xy[i]
            dx = xi - p0[0]
            dy = yi - p0[1]
            d2 = dx * dx + dy * dy
            dists.append((d2, (dx, dy)))
        # 거리 제곱 기준 오름차순 정렬해서 앞의 두 개가 인접 변
        dists.sort(key=lambda t: t[0])
        v1 = dists[0][1]  # 인접 변 1 (로컬 x 또는 y)
        v2 = dists[1][1]  # 인접 변 2 (서로 직교여야 함)

        # 벡터 길이
        l1 = math.hypot(v1[0], v1[1])
        l2 = math.hypot(v2[0], v2[1])

        if l1 == 0.0 or l2 == 0.0:
            raise ValueError("Degenerate box edges")

        # 세계 x축 (1,0)에 더 가까운 쪽을 로컬 x축으로 선택 (순서 고정 목적)
        def closeness_to_world_x(v):
            vx, vy = v
            # |cos(theta)| = |vx|/||v||, 길이로 나눠 비교해도 되지만
            # 길이가 다르므로 단순 |vx|/||v|| 값을 사용
            return abs(vx) / math.hypot(vx, vy)

        if closeness_to_world_x(v1) >= closeness_to_world_x(v2):
            x_axis_vec = v1
            y_axis_vec = v2
            lx, ly = l1, l2
        else:
            x_axis_vec = v2
            y_axis_vec = v1
            lx, ly = l2, l1

        # yaw: 로컬 x축의 각도
        yaw = math.atan2(x_axis_vec[1], x_axis_vec[0])
        # [-pi, pi)로 정규화
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

        lengths = (lx, ly, lz)
        return center, lengths, yaw

    def _parse_aabb_and_center_size(self, text):
        """
        text: EXACTLY 8 lines, each 'x,y,z' (floats).
        returns: (ok: bool, center(tuple), size(tuple))
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 8:
            return False, None, None
        lines = lines[:8]
        xs, ys, zs = [], [], []
        for i, ln in enumerate(lines, 1):
            try:
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) != 3:
                    rospy.logwarn("Line %d not 3 components: %r", i, ln)
                    return False, None, None
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                xs.append(x); ys.append(y); zs.append(z)
            except Exception as e:
                rospy.logwarn("Failed to parse line %d: %r (%s)", i, ln, e)
                return False, None, None
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)
        sx = max(xmax - xmin, 0.0)
        sy = max(ymax - ymin, 0.0)
        sz = max(zmax - zmin, 0.0)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        return True, (cx, cy, cz), (sx, sy, sz)
    
    def _make_cube_marker(self, center, size, yaw_rad=0.0, rgba=(0.1, 0.8, 0.2, 0.35), marker_id=0):
        cx, cy, cz = center
        sx, sy, sz = size
        qx, qy, qz, qw = tft.quaternion_from_euler(0.0, 0.0, yaw_rad)
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "selection"
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = cx
        m.pose.position.y = cy
        m.pose.position.z = cz
        m.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        m.scale.x = sx
        m.scale.y = sy
        m.scale.z = sz
        r, g, b, a = rgba
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        m.lifetime = rospy.Duration(0)
        return m
    
    def preprocess_sg(self, partial_sg: dict):
        """
        Scene Graph JSON을 받아 LLM 프롬프트에 사용하기 좋은 형태로 전처리
        (원본 gpt_llm.py의 로직을 그대로 이관)
        """
        out = {"objects": [], "relationships": []}
        nodes = (partial_sg or {}).get("objects", []) or []
        edges = (partial_sg or {}).get("relations", []) or []
        for node in nodes:
            out["objects"].append({
                "object_id": node.get("id"),
                "raw_label": node.get("label"),
                "center":    node.get("center"),
            })

        # out["relationships"] = edges[:self.edge_ths]
        out["relationships"] = edges
        out["robot_pose"]    = (partial_sg or {}).get("robot_pose")
        out["scene_name"]    = (partial_sg or {}).get("scene_name")
        return out

    def preprocess_rich_sg(self, partial_sg: dict):
        out = {"objects": [], "relationships": []}
        nodes = (partial_sg['regions']['0'] or {}).get("objects", []) or []
        edges = (partial_sg['regions']['0'] or {}).get("relationships", []) or []
        for node in nodes:
            out["objects"].append({
                "object_id": node.get("object_id"),
                "raw_label": node.get("raw_label"),
                "center": node.get("center"),
                "color_labels": node.get("color_labels"),
                "volume": node.get("volume"),
                "bbox": node.get("bbox"),
            })

        out["relationships"] = edges
        out["robot_pose"] = (partial_sg or {}).get("robot_pose")
        out["scene_name"] = (partial_sg or {}).get("scene_name")
        return out

    def build_scene_prompt(self, question: str, partial_sg: dict, partial_rich_sg: dict, guidance: str, output_format: str) -> str:
        """Scene Graph 정보를 포함하는 프롬프트 문자열 생성"""

        use_rich = True

        if use_rich:
            sg = partial_rich_sg
        else:
            sg = partial_sg

        scene = sg.get("scene_name", "Unknown") # scene_name 추출
        nodes = sg.get("objects", []) or []
        edges = sg.get("relationships", []) or []
        
        obj_lines = []
        for n in nodes:
            if use_rich:
                color_labels = [c for c in n.get("color_labels", []) if isinstance(c, str) and c.lower() != "n/a"]
                center = [f"{c:.4f}" for c in n.get("center", [])]
                obj_lines.append(f"{n.get('object_id')}: name={n.get('raw_label')}, color={color_labels}, center={center}, volume={n.get('volume'):.4f}")
            else:
                center = [f"{c:.4f}" for c in n.get("center", [])]
                obj_lines.append(f"{n.get('object_id')}: name={n.get('raw_label')}, center={center}")
        rel_lines = []
        for rtype, rmap in edges.items():
            filt = {}
            for k, v in rmap.items():
                k_str = str(k)
                if v:
                    filt[k_str] = v
            if filt:
                rel_lines.append(f"{rtype}:")
                for kk, vv in filt.items():
                    rel_lines.append(f"{kk}: {vv}")

        objects_str = chr(10).join(obj_lines).replace("'", "").replace('"', "")
        relationships_str = chr(10).join(rel_lines).replace("'", "").replace('"', "")

        if use_rich:
            obj_explanation = "Object id: label, color, center coordinate, volume"
        else:
            obj_explanation = "Object id: label, center coordinate"

        if self.last_qtype == 1:

            prompt = f"""\
SYSTEM:
You are an AI assistant helping a robot understand a 3D scene.
You provide multi-view 360-degree images of the environment.
Your task is to analyze the user's question to provide a precise answer in the required format.
Answer the user's request as best as possible using the provided images.

USER REQUEST:
{question}

TASK:
{guidance}

{output_format}
"""

        else:
            prompt = f"""\
SYSTEM:
You are an AI assistant helping a robot understand a 3D scene.

You will be given a scene graph with a list of objects (with indices) and their relationships.
Objects are provided in the form '{obj_explanation}'.
Relations are provided in the form 'relation: Target: [Anchor]', e.g., the relation types are defined as follows:
- Above: Target is above the anchor. Synonyms: Over.
- Below: Target is below the anchor. Synonyms: Under, Beneath, Underneath.
- Between: Target is between two anchors. Synonyms: In the middle of, Inbetween.
- Near: Target is within a threshold distance of the anchor. Synonyms: Next to, Close to, Adjacent to, Beside.

You also provide multi-view 360-degree images of the environment.
Your task is to analyze the scene graph and the user's question to provide a precise answer in the required format.
Answer the user's request as best as possible using the scene graph and the provided images.

SCENE INFORMATION:
- Scene:
OBJECTS:
{objects_str}

RELATIONSHIPS:
{relationships_str}

USER REQUEST:
{question}

TASK:
{guidance}

{output_format}
"""

        return prompt.strip()


def main():
    try:
        LLMPromptBuilder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
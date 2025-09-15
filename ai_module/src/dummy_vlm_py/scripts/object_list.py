#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json, math
from typing import Dict, Any, List, Optional
from std_msgs.msg import String
import os

##json 파일을 publish하는 것 뿐만 아니라 경로에 저장까지 하는 함구


##heading을 구하는 함수로 SORT3D와 같은 방법으로 구함. 직육면체임을 가정하고 상면의 YAW를 구해서 HEADING에 저장하는 방식
def _yaw_from_bbox(bbox: List[List[float]]) -> Optional[float]:
    """
    8개 꼭짓점으로 주어진 3D bbox(상면 4점 + 하면 4점)에서
    상면의 가장 긴 에지 방향으로 yaw(라디안)를 근사 계산.
    bbox: [[x,y,z], ...] 길이>=4 가정.
    """
    if not bbox or len(bbox) < 4:
        return None
    p = bbox[:4]  # 상면 4점 가정
    edges = [
        (p[1][0]-p[0][0], p[1][1]-p[0][1]),
        (p[2][0]-p[1][0], p[2][1]-p[1][1]),
        (p[3][0]-p[2][0], p[3][1]-p[2][1]),
        (p[0][0]-p[3][0], p[0][1]-p[3][1]),
    ]
    ex, ey = max(edges, key=lambda e: (e[0]**2 + e[1]**2))
    if ex == 0 and ey == 0:
        return None
    return math.atan2(ey, ex)  # 라디안

##bbox의 size를 계산하는 코드로써 마찬가지로 제공되어지지만, 없으면 bbox min과 max의 차이로 구현
def _safe_get_size(obj: Dict[str, Any]) -> Optional[List[float]]:
    """size가 없으면 bbox_max - bbox_min 로 보정"""
    if "size" in obj and obj["size"] is not None:
        return list(obj["size"])
    if "bbox_min" in obj and "bbox_max" in obj:
        mn, mx = obj["bbox_min"], obj["bbox_max"]
        if mn and mx and len(mn) == 3 and len(mx) == 3:
            return [mx[i]-mn[i] for i in range(3)]
    return None

##object의 center를 계산하는 코드  partial scene graph에서 center 좌표를 제공하지만 만약에 없는 경우에는 bbox의 min,max 의 평균 값으로 선정
def _safe_get_center(obj: Dict[str, Any]) -> Optional[List[float]]:
    """center가 없으면 (bbox_min + bbox_max)/2 로 보정"""
    if "center" in obj and obj["center"] is not None:
        return list(obj["center"])
    if "bbox_min" in obj and "bbox_max" in obj:
        mn, mx = obj["bbox_min"], obj["bbox_max"]
        if mn and mx and len(mn) == 3 and len(mx) == 3:
            return [(mn[i]+mx[i])/2.0 for i in range(3)]
    return None

## 색상 퍼센트 파서 + 최대 비율 색 추출 (SG의 color_labels / color_percentages 사용)
def _parse_pct(v) -> Optional[float]:
    """
    "34.2", "34.2%", 34.2, 0.342 같은 입력을 모두 퍼센트[0~100]로 통일
    """
    if v is None:
        return None
    try:
        s = str(v).strip().lower()
        if s.endswith('%'):
            s = s[:-1]
        val = float(s)
        # 0~1 스케일이면 0~100 환산
        if 0.0 <= val <= 1.0:
            val *= 100.0
        return val
    except Exception:
        return None

def _pick_top_color_from(obj: Dict[str, Any], min_pct: float = 0.0) -> Optional[str]:
    """
    SG의 color_labels / color_percentages에서 가장 비율이 큰 색을 선택.
    - 'N/A'는 제외
    - 퍼센트가 전혀 없으면 원래 순서를 유지(첫 유효 라벨 사용)
    - min_pct로 너무 작은 비중은 제외 가능(기본 0.0: 무조건 채택)
    """
    labels = obj.get("color_labels") or []
    pcts = obj.get("color_percentages") or []
    candidates = []
    for i, lab in enumerate(labels):
        if not lab or str(lab).lower() == "n/a":
            continue
        pct = _parse_pct(pcts[i]) if i < len(pcts) else None
        candidates.append((str(lab).strip().lower(), pct))

    if not candidates:
        return None

    # pct=None 은 -inf 취급 → 뒤로 밀림, 전부 None이면 원래 순서 유지(파이썬 정렬 안정성)
    candidates.sort(key=lambda x: (float('-inf') if x[1] is None else x[1]), reverse=True)
    top_lab, top_pct = candidates[0]

    if top_pct is not None and top_pct < min_pct:
        return None
    return top_lab


##가공 부분에 해당하는 함수
def build_response_from_partial_sg(sg_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    partial_sg(JSON dict)를 받아 다음 형태로 가공:
    {
      "scene_name": ...,
      "timestamp_ros": ...,
      "robot_pose": {...},
      "response": {
        <obj_id>: {
          "object_id": <int>,              # 내부에도 object_id 포함
          "name": <str>,                   # nyu_label 우선, 없으면 raw_label
          "centroid": [x,y,z],             # center 또는 (bbox_min+bbox_max)/2
          "caption": null,                 # partial_sg엔 없으므로 None
          "dimensions": [l,w,h],           # size 또는 bbox_max-bbox_min
          "heading": <float or null>,      # bbox 상면 최장 에지 방향(rad)
          "largest_face": <float or null>  # max(l*w, w*h, l*h)
        },
        ...
      }
    }
    """
    response: Dict[int, Dict[str, Any]] = {}
    objects = sg_json.get("objects", []) or []
    for o in objects:
        try:
            oid = int(o.get("object_id"))
        except Exception:
            continue
        name = o.get("nyu_label") or o.get("raw_label") or "unknown"
        centroid = _safe_get_center(o)
        size = _safe_get_size(o)
        heading = None
        if "bbox" in o and o["bbox"]:
            heading = _yaw_from_bbox(o["bbox"])
        largest_face = None
        ##largest face 계산하는 코드로 bbox에서 가장 큰 면의 넓이를 의미
        if size and len(size) == 3:
            l, w, h = size
            largest_face = max(l*w, w*h, l*h)

        ## [변경] caption 생성: 색 정보 + 이름 조합("black chair" 등)
        color = _pick_top_color_from(o, min_pct=0.0)  # 필요 시 0.15 등으로 임계치 조정
        if color and name and name != "unknown":
            caption = f"{color} {name}"
        elif name and name != "unknown":
            caption = name
        elif color:
            caption = color
        else:
            caption = None

        response[oid] = {
            "object_id": oid,
            "name": name,
            "centroid": centroid,
            "caption": caption,  # ← 기존 None에서 변경
            "dimensions": size,
            "heading": heading,
            "largest_face": largest_face,
        }
    # partial scene graph에서의 앞 부분에 해당하는 부분은 그냥 그대로 사용
    out = {
        "scene_name": sg_json.get("scene_name"),
        "timestamp_ros": sg_json.get("timestamp_ros"),
        "robot_pose": sg_json.get("robot_pose"),
        "response": response
    }
    return out

class PartialSGAdapter(object):
    """
    - 입력: partial_sg_generator가 퍼블리시한 partial SG (std_msgs/String, JSON)를 구독
    - 처리: partial scene graph를 object list형식으로 가공하기
    - 출력: 가공된 JSON을 다시 퍼블리시
    """
    def __init__(self):
        ##!중요 generalize 다른 곳에서도 돌아가게 하기 위해 손대야할 부분
        self.scene_name = "livingroom_1"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.abspath(os.path.join(script_dir, "../../../../"))
        self.save_dir = os.path.join(self.base_path, "collected_data", self.scene_name, "object_list")


        ##이 부분이 partial_scene_graph를 구독하는 부분
        rospy.init_node("partial_sg_adapter", anonymous=True)
        default_in = "/partial_scene_graph_generator/partial_scene_graph"  # "~partial_scene_graph"의 fully-qualified 예시
        self.in_topic  = rospy.get_param("~in_topic", default_in)
        self.out_topic = rospy.get_param("~out_topic", "/processed_partial_sg")

        self.pub = rospy.Publisher(self.out_topic, String, queue_size=10, latch=True)
        self.sub = rospy.Subscriber(self.in_topic, String, self.cb, queue_size=10)
        os.makedirs(self.save_dir, exist_ok=True)
    #call back 함수로써 scenegraph가 발행되고 이 신호를 받을 때마다 실행되는 함수
    def cb(self, msg: String):
        ##입력
        rospy.loginfo("reached")
        try:
            sg = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[partial_sg_adapter] JSON parse error: {e}")
            return
        ##가공
        try:
            out=build_response_from_partial_sg(sg)
        except Exception as e:
            rospy.logwarn(f"[partial_sg_adapter] build failed: {e}")
            return
        ##출력
        try:
            rospy.loginfo("reached")
            scene = out.get("scene_name") or "scene"
            ts = out.get("timestamp_ros") or rospy.Time.now().to_sec()
            # 파일명 안전 문자만 남기기
            safe_scene = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(scene))
            fname = f"processed_object_list_{safe_scene}_{ts:.6f}.json"
            fpath = os.path.join(self.save_dir, fname)

            # 원자적 저장
            tmp = fpath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            os.replace(tmp, fpath)

            rospy.loginfo(f"[partial_sg_adapter] saved to {fpath}")
        except Exception as e:
            rospy.logwarn(f"[partial_sg_adapter] save failed: {e}")
        self.pub.publish(String(data=json.dumps(out, ensure_ascii=False)))

if __name__ == "__main__":
    try:
        node = PartialSGAdapter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

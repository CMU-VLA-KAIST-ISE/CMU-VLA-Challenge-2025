#!/opt/conda/envs/python310/bin/python3
# -*- coding: utf-8 -*-

"""
Fusion One-Shot + Rich + Merge Scene Graph Node (Full)
- /instruction 들어오면 1회 파이프라인 실행
- "정지 상태 지속" 자동 감지 후 1회 자동 실행(auto_on_stop)
- Odometry 기반 + TF 기반(폴백) 정지 감지 모두 지원
- /fusion/scene_graph, /fusion/scene_graph_rich : 스냅샷 SG
- /fusion/scene_graph_merged, /fusion/scene_graph_rich_merged : 누적 SG
- /fusion/segments : 세그 출력
- /fusion/markers(MarkerArray, latched), /fusion/centroids_map(PoseArray, latched)
- 프롬프트 덮어쓰기 방지 (busy 우선, 그 다음 cooldown)
- 세그멘테이션 타임아웃 + 워치독으로 is_busy 영구 고착 방지
- TF lookup 폴백(요청시각 실패 시 최신 시각으로 재시도)으로 extrapolation 경고 완화
"""

from __future__ import annotations
import concurrent.futures as cf
import copy
import queue

import os, io, re, json, time, base64, binascii, threading, math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft

from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_msgs.msg import String, Empty, Bool, Int32
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, PoseStamped, Pose2D
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw
from visualization_msgs.msg import Marker, MarkerArray

from scene_graph_builder import build_scene_graph, SceneGraphParams
from collections import deque

# -------- Gemini GenAI (google-genai)
from google import genai
#import google.generativeai as genai
from google.genai import types

# ----------------- small utils -----------------
DEFAULT_CAMERA = dict(width=1920, height=640, hfov_deg=360.0, vfov_deg=120.0)
DEFAULT_L2C   = dict(x=-0.12, y=-0.075, z=0.265, roll=-1.5707963, pitch=0.0, yaw=-1.5707963)

def now_ts_str():
    return time.strftime("%Y%m%d_%H%M%S_") + f"{int((time.time()%1)*1e6):06d}"

def decode_data_url_to_bytes(s: str) -> bytes:
    if not s:
        return b""
    if s.startswith("data:"):
        comma = s.find(",")
        if comma != -1:
            s = s[comma+1:]
    s = re.sub(r"\s+", "", s).replace("-", "+").replace("_", "/")
    pad = (-len(s)) % 4
    if pad: s += "=" * pad
    try:
        return base64.b64decode(s, validate=True)
    except binascii.Error:
        s2 = re.sub(r"[^A-Za-z0-9+/=]", "", s)
        pad2 = (-len(s2)) % 4
        if pad2: s2 += "=" * pad2
        return base64.b64decode(s2)
def save_xyz_to_ply_ascii(out_path: str, xyz: np.ndarray, rgb_val=(128, 128, 128)):
    """
    xyz: (N,3) float32/float64, 카메라 프레임 좌표 등
    rgb_val: 모든 포인트에 동일 적용할 RGB (0~255)
    """
    import os
    xyz = np.asarray(xyz, dtype=np.float32)
    # NaN/Inf 제거
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    N = int(xyz.shape[0])

    r, g, b = [int(max(0, min(255, c))) for c in rgb_val]

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header)
        # 한 줄: x y z r g b
        arr = np.c_[xyz,
                    np.full((N, 1), r, dtype=np.uint8),
                    np.full((N, 1), g, dtype=np.uint8),
                    np.full((N, 1), b, dtype=np.uint8)]
        np.savetxt(f, arr, fmt="%.5f %.5f %.5f %d %d %d")

def cloud2_to_xyz_array(msg: PointCloud2) -> Tuple[np.ndarray, str, rospy.Time]:
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if len(pts) == 0:
        return np.zeros((0,3), dtype=np.float32), (msg.header.frame_id or ""), (msg.header.stamp or rospy.Time(0))
    arr = np.asarray(pts, dtype=np.float32)
    return arr, (msg.header.frame_id or ""), (msg.header.stamp or rospy.Time(0))

def scan2pixels_mecanum(laserCloud: np.ndarray, CAM: dict, LIDAR: dict) -> np.ndarray:
    cam_offset = np.array([CAM["x"], CAM["y"], CAM["z"]], dtype=np.float32)
    xyz = laserCloud[:, :3].astype(np.float32)
    xyz = xyz - cam_offset
    W, H = int(CAM["width"]), int(CAM["height"])
    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) + 1e-9
    u = (W / (2 * np.pi)*np.arctan2(-xyz[:,1], xyz[:, 0])+W/2)
    v = (H / (2 * np.pi /3)*np.arctan(-xyz[:, 2] / (horiDis))+H/2)
    min_u_raw = int(np.min(u)); max_u_raw = int(np.max(u))
    min_v_raw = int(np.min(v)); max_v_raw = int(np.max(v))
    return np.stack([u, v, horiDis.astype(np.float32)], axis=1)

def write_pcd_float32(filepath, pts: np.ndarray, field_names):
    pts = np.asarray(pts, dtype=np.float32, order="C")
    K = pts.shape[1]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(field_names)}\n"
        f"SIZE {' '.join(['4'] * K)}\n"
        f"TYPE {' '.join(['F'] * K)}\n"
        f"COUNT {' '.join(['1'] * K)}\n"
        f"WIDTH {pts.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {pts.shape[0]}\n"
        "DATA binary\n"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes(order="C"))

def draw_points_overlay(pil_img: PILImage.Image, u: np.ndarray, v: np.ndarray,
                        pick_mask: Optional[np.ndarray], out_path: str,
                        max_points: int = 80000):
    W, H = pil_img.size
    idx_all = np.arange(u.shape[0])
    if u.shape[0] > max_points:
        idx_all = np.random.choice(idx_all, size=max_points, replace=False)
    base = pil_img.convert("RGB").copy()
    dr = ImageDraw.Draw(base)
    for i in idx_all[: max_points//2]:
        x, y = int(u[i]), int(v[i])
        dr.point((x, y), fill=(180,180,180))
    if pick_mask is not None and pick_mask.any():
        idx_pick = np.where(pick_mask)[0]
        if idx_pick.size > max_points//2:
            idx_pick = np.random.choice(idx_pick, size=max_points//2, replace=False)
        for i in idx_pick:
            x, y = int(u[i]), int(v[i])
            dr.point((x, y), fill=(255,32,32))
    base.save(out_path, "PNG")

def _aabb_to_corners(mins, maxs):
    x0,y0,z0 = mins; x1,y1,z1 = maxs
    return [
        [x0, y1, z1], [x1, y1, z1], [x1, y0, z1], [x0, y0, z1],
        [x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0],
    ]

COLOR_NAMES = ["black","white","gray","red","orange","yellow","green","cyan","blue","purple","brown"]

def _rgb_to_hsv_np(rgb_uint8: np.ndarray):
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    diff = cmax - cmin + 1e-12
    h = np.zeros_like(cmax)
    mask = (cmax == r)
    h[mask] = (60 * ((g[mask]-b[mask]) / diff[mask]) + 360) % 360
    mask = (cmax == g)
    h[mask] = (60 * ((b[mask]-r[mask]) / diff[mask]) + 120) % 360
    mask = (cmax == b)
    h[mask] = (60 * ((r[mask]-g[mask]) / diff[mask]) + 240) % 360
    s = np.where(cmax == 0, 0.0, diff / (cmax + 1e-12))
    v = cmax
    return h, s, v

def _classify_basic_colors(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    idx = np.zeros_like(h, dtype=np.int32)
    black = (v < 0.20)
    white = (v > 0.90) & (s < 0.10)
    gray  = (s < 0.20) & (~black) & (~white)
    idx[black] = COLOR_NAMES.index("black")
    idx[white] = COLOR_NAMES.index("white")
    idx[gray]  = COLOR_NAMES.index("gray")

    chroma = ~(black | white | gray)
    red    = ((h <= 15) | (h >= 345)) & chroma
    orange = (h > 15) & (h <= 45) & chroma
    yellow = (h > 45) & (h <= 65) & chroma
    green  = (h > 65) & (h <= 170) & chroma
    cyan   = (h > 170)& (h <= 200) & chroma
    blue   = (h > 200)& (h <= 255) & chroma
    purple = (h > 255)& (h <= 290) & chroma
    brown  = ((h > 15) & (h <= 45) & (v < 0.6) & (s > 0.2)) & chroma

    idx[red]    = COLOR_NAMES.index("red")
    idx[orange] = COLOR_NAMES.index("orange")
    idx[yellow] = COLOR_NAMES.index("yellow")
    idx[green]  = COLOR_NAMES.index("green")
    idx[cyan]   = COLOR_NAMES.index("cyan")
    idx[blue]   = COLOR_NAMES.index("blue")
    idx[purple] = COLOR_NAMES.index("purple")
    idx[brown]  = COLOR_NAMES.index("brown")
    return idx

def _top3_color_stats(rgb_pixels_uint8: np.ndarray):
    if rgb_pixels_uint8 is None or rgb_pixels_uint8.size == 0:
        return [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]], ["N/A","N/A","N/A"], [0.0,0.0,0.0], False
    N = rgb_pixels_uint8.shape[0]
    if N > 20000:
        sel = np.random.choice(np.arange(N), size=20000, replace=False)
        rgb = rgb_pixels_uint8[sel]
    else:
        rgb = rgb_pixels_uint8
    h,s,v = _rgb_to_hsv_np(rgb)
    cat_idx = _classify_basic_colors(h,s,v)
    uniq, counts = np.unique(cat_idx, return_counts=True)
    order = np.argsort(counts)[::-1]
    uniq = uniq[order]; counts = counts[order]
    total = float(rgb.shape[0] + 1e-12)
    vals = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
    labels = ["N/A","N/A","N/A"]
    percs = [0.0,0.0,0.0]
    for i in range(min(3, uniq.size)):
        k = uniq[i]
        mask = (cat_idx == k)
        mean_rgb = rgb[mask].mean(axis=0)
        vals[i] = [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])]
        labels[i] = COLOR_NAMES[int(k)]
        percs[i] = float(counts[i] / total)
    return vals, labels, percs, True

def _normalize_color_fields(seg: Dict[str,Any]):
    vals = seg.get("color_vals")
    labels = seg.get("color_labels")
    percs = seg.get("color_percentages")
    if not (isinstance(vals, list) and len(vals) == 3):
        vals = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
    if not (isinstance(labels, list) and len(labels) == 3):
        labels = ["N/A","N/A","N/A"]
    out_percs = [0.0,0.0,0.0]
    if isinstance(percs, list) and len(percs) == 3:
        for i,p in enumerate(percs):
            try: out_percs[i] = float(p)
            except: out_percs[i] = 0.0
    return vals, labels, out_percs, any(x>0 for x in out_percs)

# ----------------- Segmenter wrapper -----------------
@dataclass
class SegItem:
    box_2d: List[int]
    mask: str
    label: str

@dataclass
class RelevanceScore(dict):
    information_completeness: int
    object_coverage: int
    coverage_extent: int

ob_list = None
class GeminiSegmenter:
    def __init__(self, model="gemini-2.5-flash"):
        api_key = rospy.get_param("~gemini_api_key")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        rospy.Subscriber("/challenge_question", String, self.on_question, queue_size=1)
        self.challenge_question = ""
        # self.rel_score_pub=rospy.Publisher("/rel_score",Int32,queue_size=1)  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        self.last_rel_score = None  # gpt: 최근 rel_score 저장 (파일명에 사용)

    def on_question(self, msg: String):
        self.challenge_question = msg.data

    def run_once(self, pil_image: PILImage.Image, prompt: Optional[str]=None):
        global ob_list
        if ob_list==None:
            prompt_txt = f"""Extract object names from QUESTION
Rules:
- Output ONLY the words (no extra text).
- Lowercase, singular form, unique.
- Comma-separated with space(e.g., sofa, window).
QUESTION:
{self.challenge_question}"""
            resp1 = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt_txt]
            )
            ob_list=getattr(resp1, "text", "") or ""
            rospy.loginfo(f"object list : {ob_list}")
        if pil_image is None:
            return [], None, None
        im = pil_image.copy()
        im.thumbnail([1024, 1024], PILImage.Resampling.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="PNG")
        image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        rospy.loginfo(f"[fusion] question : {self.challenge_question}")
        # OBJECTS_UNITY = ['display ledge', 'paper box', 'notice board', 'stove', 'tree', 'soap', 'wall shelf', 'lotion', 'bag', 'calligraphy painting', 'headphones', 'drawing', 'soccer ball', 'exterior structure', 'vending machine', 'easel', 'door glass segment', 'tv stand with boxes', 'wall with squares and triangles', 'TV cabinet', 'side table', 'panel frame', 'plate', 'photo', 'symbol decoraion', 'Arabic jar', 'bed bench', 'terrain', 'bowl', 'standing lamp', 'cup', 'wardrobe', 'fan decoration',
        #              'fire hose', 'unknown', 'potted branch', 'bed', 'desk table', 'bird decoration', 'carpet', 'window', 'block glass angled', 'paint', 'curtain', 'computer monitor', 'outside door', 'tatami', 'kitchen island', 'cube', 'block wall', 'canvas', 'tv stand', 'lantern', 'shower head', 'elephant decoration', 'office double desk', 'bowl of apples', 'oven', 'trophy decoration', 'kitchen counter', 'lamps', 'folders', 'bench', 'sink cabinet', 'countyard ground',
        #              'wooden floor/wall/ceiling panel', 'elephant figurine', 'frame', 'knife', 'candle', 'bathroom walls', 'sink', 'blanket', 'tv remote', 'tap', 'stool', 'tray papers', 'kitchen cabinet', 'wine glass', 'computer', 'cupboard', 'toilet glass', 'studio speaker', 'lamp', 'oven door', 'marker', 'floor side', 'clothes', 'kettle', 'perfume bottle', 'fire alarm', 'gymnastics decoration', 'dish', 'desk cabinet', 'build block', 'map wall decal', 'eye glasses', 'dvd', 'potted cactus',
        #              'kitchen door', 'bedside table', 'beer bottle', 'water tank', 'refridgerator', 'pyramid candle holder', 'phone', 'ring l fix', 'partition wall', 'kitchenette', 'chair', 'nightstand', 'fireplace', 'cooker', 'box', 'tablecloth', 'water cooler', 'block glass', 'sphere decoration', 'structure', 'video beam straight', 'eraser', 'paper holder', 'block simple 4m', 'air conditioner', 'floor', 'reception desk', 'pen holder', 'soap dish', 'tray', 'drawer', 'projector screen', 'books',
        #              'cabinet', 'fire extinguisher pipes', 'monitor', 'balcony door', 'sake bottle', 'fence', 'block glass framed', 'cable', 'block glass elevator', 'coffee table', 'computer mouse', 'zen stone decoration', 'trashcan', 'circular light', 'firewood', 'fotel', 'potted bamboo', 'entrance door', 'fridge drinks', 'coffee cup', 'jar', 'file', 'scanner', 'dice decoration', 'glass', 'ottoman', 'shoe rack', 'sticky notes', 'hanger', 'ceiling', 'kitchen knife', 'mouse', 'guitar', 'spoon',
        #              'suitcase', 'notebook', 'sculpture', 'bed frame', 'door frame', 'welcome desk', 'pillow', 'wall tree', 'soda can', 'soap bottle', 'elevator', 'evergreen', 'flower', 'flower box', 'conference table', 'dvd player', 'wash base', 'balcony railing', 'spot light', 'kitchen bar', 'sofa pillows', 'juice', 'card', 'couch', 'handle', 'light switch', 'placemat', 'build glass', 'decorative ball', 'can of coke', 'folding screen', 'reading light', 'toilet', 'celling', 'notecards', 'desk',
        #              'utensils', 'evergreen ivy', 'pencils mug', 'espresso machine', 'computer case', 'fork', 'speakers', 'wall mozaic', 'door frame ', 'coffee machine', 'teapot', 'calendar', 'hand dryer', 'towel rail', 'bathroom cabinet', 'ash tray', 'dining chair', 'wall lamp', 'build block door ', 'fossil decoration', 'mirror', 'spice jar', 'newspaper', 'building', 'stepstool', 'block glass 4m', 'candlestick', 'sushi', 'doorframe', 'recycle bin', 'sauce bowl', 'speaker', 'pot', 'beer glass',
        #              'elevator arrows', 'organizer', 'tv cabinet', 'bookcase', 'table', 'range hood', 'kitchen door frame', 'security camera', 'sofa cushion', 'office chair', 'exterior walls', 'floor middle', 'curtains', 'elevate box', 'dining table', 'shoes', 'desk light', 'tv bench', 'exit sign', 'plant stand', 'oil bottle', 'book', 'document folder', 'gate', 'coffee pot', 'chess', 'poster', 'ceiling light', 'mouse pad', 'night stand', 'bottle', 'framed record', 'clock', 'water bottle',
        #              'fridge for drinks', 'camera', 'buddha decoration', 'window frame', 'shell fotel', 'map', 'folder', 'bread', 'ball candle holder', 'hat', 'arc', 'manager chair', 'tea table', 'knife rack', 'face cream', 'trash can', 'dracaena', 'focus light', 'bathtub', 'files', 'palm', 'mattress', 'tower decoration', 'wall decal', 'paper cup', 'whiteboard', 'glass door', 'file cabinet', 'fire extinguisher box', 'ceiling lamp', 'lipstick', 'quilt', 'column', 'towel rack', 'switch',
        #              'angled sofa', 'hookah wire', 'shelf', 'paper', 'wall', 'coffe cup plate', 'crystal ball decoration', 'sofa', 'welcome sign', 'glass wall', 'shower tap', 'hookah', 'otherstructure', 'boot', 'drawers', 'dumbbell', 'toilet paper', 'pen', 'ashtray', 'wine bottle', 'shower', 'cutting board', 'fridge', 'ports', 'bar chair', 'horse figurine', 'round table', 'wardrobe door', 'magazine', 'plant', 'newtons cradle', 'air vent', 'shelves', 'towel', 'bedroom light', 'wifi',
        #              'video beam', 'plane', 'laptop', 'dressing table', 'microwave', 'tv', 'windows', 'stair', 'keyboard', 'vase', 'trash bin', 'panel poster', 'sliced bread', 'water pro daytime', 'potted plant', 'pillows', 'circle decoration', 'floor angle', 'sign', 'wooden block', 'picture', 'office printer', 'painting', 'round box', 'chopsticks', 'projector', 'basket', 'door', 'dressing chair', 'umbrella', 'stairs', 'pottery', 'flowers', 'slipper', 'deck chair', 'soap holder',
        #              'kitchen light', 'elevator block']
        # _prompt = (
        #     "You are an expert segmentation model for mobile robots.\n"
        #     f"Your ONLY task is to segment {ob_list}"
        #     "\n"
        #     "RUle:\n"
        #     "Return JSON array with objects, each has EXACT KEYS:\n"
        #     "- box_2d: [y0,x0,y1,x1] in 0..1000\n"
        #     "- mask: base64 PNG (any size)\n"
        #     "- if you find objects whose label is one of the following, put that label. (e.g. you find couch then label it sofa.) Otherwise, you can freely label what ever you see.\n"
        #     # f"- label: one of {', '.join(OBJECTS_UNITY)}\n"
        #     f"- label: one of {', '.join(ob_list)}\n"
        #     "- Never output the literal string 'label' as the label value."
        #     "- Proceed with segmentation only if your confidence is at least 90%."
        #     "- If you find nothing relevant to the question, return an empty array [] only.\n"
        # )
        _prompt = f'Segment objects in the image that in {ob_list} (up to strictly 5)'
        
        # (
        #     "You are an expert segmentation model for mobile robots.\n"
        #     f"Allowed labels (segment ONLY these): {ob_list}\n"
        #     "Rules:\n"
        #     f"- Proceed with segmentation for an object ONLY if your confidence for that object is in {ob_list}>= 90%.\n"
        #     "- Output a JSON array of objects. Each object MUST have EXACTLY these keys (no others):\n"
        #     "  - box_2d: [y0,x0,y1,x1] with integers in 0..1000 (ensure y0<y1 and x0<x1)\n"
        #     "  - mask: base64 PNG (any size)\n"
        #     f"  - label: only one from {ob_list}\n"
        #     "- If no qualifying objects are found, output 'there are no allowed labels'\n"
        #     "Return ONLY the JSON array. No explanations, no code fences."
        # )
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=list[SegItem],  # keep existing schema usage
        )
        resp = self.client.models.generate_content(model=self.model_name, contents=[_prompt, image_part], config=cfg)

        _prompt_3 = f'''
You are an evaluation model.  
Your task is to assign a **relevance_score** for a given Image with respect to a Challenge Question and an Object List.  
IMPORTANT: Always evaluate **as conservatively as possible**. Be strict and assign lower scores unless the evidence is very strong. Never give a high score unless it is absolutely justified.  

You must assess multiple dimensions of relevance, each on a 0-10 scale, and output them in JSON format.  
Do not directly average them—just return the individual scores.  

### Inputs:  
- **Image:** A 360° simulation environment image.  
- **Challenge Question:** A counting-related question requiring certain objects.  
- **Object List:** A set of objects relevant to solving the challenge question.  

### Evaluation Criteria (all 0-10 scale, described in detail):  

1. **information_completeness** (confidence + risk of being wrong):  
   - Consider whether the image contains enough information to answer the challenge question with confidence.  
   - High scores: The image alone provides nearly all information needed with extremely high confidence, and there is little or no risk of hidden/occluded objects.  
   - Medium scores: The image provides partial or uncertain information; there is some risk of missing/hidden objects that could change the answer.  
   - Low scores: The image does not provide enough information to answer the question, or the risk of being wrong is too high.  

2. **object_coverage** (presence + occlusion/clutter):  
   - Consider how many of the required objects from the object list are present and how confidently they can be identified without occlusion.  
   - High scores: All required objects are present and clearly visible, with high confidence that none are occluded or hidden.  
   - Medium scores: All required objects appear present, but with some chance or likelihood of occlusion/clutter; or only a subset of required objects are confidently visible.  
   - Low scores: Few or none of the required objects are visible, or occlusion is so severe that coverage cannot be trusted.  

3. **coverage_extent** (proportion of image occupied by relevant area):  
   - This score represents how much of the total image is taken up by the region that directly corresponds to the challenge question.  
   - Use a **0-10 scale**, where the score maps directly to the approximate percentage of the image covered by the relevant area.  
   - 10 = The entire image is dedicated to the relevant area (100%).  
   - 5 = The relevant area takes up about half of the image (50%). 
   - 0 = The relevant area is not present in the image (0%). 

   - Intermediate values should be chosen proportionally (e.g., if the area takes about 10-15% of the image, score ≈ 1 or 2).  
   - Be conservative: if unrelated objects dominate most of the image, assign a lower score even if the relevant area is visible.

Challenge Question: {self.challenge_question}
Object List: {ob_list}
Image: <provided separately as image part>
        '''

        cfg3 = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=RelevanceScore,
            temperature=0.1
        )
        resp3 = self.client.models.generate_content(model=self.model_name, contents=[_prompt_3, image_part], config=cfg3)
        resp3 = getattr(resp3, "text", "")
        print(f"[fusion] resp3 text: {resp3}")
        resp3 = json.loads(resp3)
        rel_score = 0
        for key, item in resp3.items():
            try:
                rel_score += int(item)
                print(f"[fusion] {key} : {item}")
            except Exception:
                print(f"[fusion] {key} parse error")
                pass
        if rel_score !=0:
            rel_score = rel_score / 3
            rel_score = round(rel_score)
        else:
            rel_score = 0

        rospy.loginfo(f"[fusion] Gemini responded rel_score: {rel_score}...")
        # rel_score=getattr(resp3, "text", "") or ""
        try:
            rel_score = int(rel_score)
        except Exception:
            rospy.logwarn(f"[Fusion] rel_score parse failed; got '{rel_score}'")
            rel_score = None
        # self.rel_score_pub.publish(Int32(rel_score))  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        self.last_rel_score = rel_score
        items = []
        try:
            items = resp.parsed or []
        except Exception:
            rospy.loginfo(f"[fusion] Gemini parse error")
            pass
        if not items:
            # fallback: try to parse JSON from text
            m = re.search(r"```json\s*(.*?)\s*```", getattr(resp, "text", ""), flags=re.S)
            raw = json.loads(m.group(1) if m else getattr(resp, "text", "[]"))
            items = [SegItem(**it) for it in raw]
        else:
            rospy.loginfo(f"[fusion] Gemini parsed {len(items)} items")
        return items, getattr(resp, "text", None), rel_score

# ----------------- Merging -----------------
_NYU_MAP = {
    "chair":        {"nyu_id":"62","nyu40_id":"5","nyu_label":"chair","nyu40_label":"chair"},
    "sofa":         {"nyu_id":"83","nyu40_id":"6","nyu_label":"sofa","nyu40_label":"sofa"},
    "table":        {"nyu_id":"61","nyu40_id":"9","nyu_label":"table","nyu40_label":"table"},
    "tv":           {"nyu_id":"71","nyu40_id":"15","nyu_label":"tv","nyu40_label":"tv"},
    "tv stand":     {"nyu_id":"-1","nyu40_id":"-1","nyu_label":"tv stand","nyu40_label":"tv stand"},
    "potted plant": {"nyu_id":"64","nyu40_id":"12","nyu_label":"plant","nyu40_label":"plant"},
    "plant":        {"nyu_id":"64","nyu40_id":"12","nyu_label":"plant","nyu40_label":"plant"},
    "desk":         {"nyu_id":"57","nyu40_id":"9","nyu_label":"desk","nyu40_label":"table"},
    "cabinet":      {"nyu_id":"65","nyu40_id":"8","nyu_label":"cabinet","nyu40_label":"cabinet"},
    "shelf":        {"nyu_id":"67","nyu40_id":"7","nyu_label":"shelf","nyu40_label":"shelf"},
    "bed":          {"nyu_id":"64","nyu40_id":"4","nyu_label":"bed","nyu40_label":"bed"},
    "monitor":      {"nyu_id":"71","nyu40_id":"15","nyu_label":"monitor","nyu40_label":"tv"},
    "books":        {"nyu_id":"96","nyu40_id":"23","nyu_label":"books","nyu40_label":"objects"},
    "door":         {"nyu_id":"12","nyu40_id":"3","nyu_label":"door","nyu40_label":"door"},
    "window":       {"nyu_id":"11","nyu40_id":"2","nyu_label":"window","nyu40_label":"window"},
    "unknown":      {"nyu_id":"-1","nyu40_id":"-1","nyu_label":"unknown","nyu40_label":"unknown"},
}
def _nyu_lookup(label):
    key = (label or "").lower().strip()
    return _NYU_MAP.get(key, {"nyu_id":"-1","nyu40_id":"-1","nyu_label":key,"nyu40_label":key})
def _safe_label(lbl: str) -> str:
    s = (lbl or "").strip().lower()
    if s in ("", "label", "object", "item"): return "unknown"
    return s

def _rect_iou(a_xyxy, b_xyxy):
    ax0,ay0,ax1,ay1 = a_xyxy
    bx0,by0,bx1,by1 = b_xyxy
    ix0 = max(ax0,bx0); iy0 = max(ay0,by0)
    ix1 = min(ax1,bx1); iy1 = min(ay1,by1)
    iw = max(0.0, ix1-ix0); ih = max(0.0, iy1-iy0)
    inter = iw*ih
    area_a = max(0.0, ax1-ax0) * max(0.0, ay1-ay0)
    area_b = max(0.0, bx1-bx0) * max(0.0, by1-by0)
    union = area_a + area_b - inter + 1e-12
    return inter/union
def _bbox8_to_xyxy(b8):
    xs = [p[0] for p in b8]; ys=[p[1] for p in b8]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

class SGMerger(object):
    def __init__(self, iou_thresh=0.25, center_dist=0.75):
        self.iou_thresh = float(iou_thresh)
        self.center_dist = float(center_dist)
        self._next_id = 0
        self.tracks: Dict[int, Dict[str,Any]] = {}
    def _new_id(self) -> int:
        i = self._next_id; self._next_id += 1
        return i
    def update_with_rich(self, rich: Dict[str,Any], stamp: float):
        if not rich: return
        objs = rich.get("regions",{}).get("0",{}).get("objects",[])
        for o in objs:
            label = _safe_label(o.get("raw_label","unknown"))
            bbox8 = o.get("bbox", None)
            center = o.get("center", None)
            if not (bbox8 and center):
                continue
            xyxy = _bbox8_to_xyxy(bbox8)
            cx,cy,cz = float(center[0]), float(center[1]), float(center[2])
            aabb = None
            if bbox8:
                xs=[p[0] for p in bbox8]; ys=[p[1] for p in bbox8]; zs=[p[2] for p in bbox8]
                aabb = {"min":[min(xs),min(ys),min(zs)], "max":[max(xs),max(ys),max(zs)]}
            best_id = None; best_score = -1.0
            for tid, tk in self.tracks.items():
                if tk["label"] != label:
                    continue
                iou = _rect_iou(xyxy, tk["xyxy"])
                if iou < self.iou_thresh:
                    continue
                d = math.hypot(cx - tk["center"][0], cy - tk["center"][1])
                if d > self.center_dist:
                    continue
                score = iou - 0.1*d
                if score > best_score:
                    best_score = score; best_id = tid
            if best_id is None:
                tid = self._new_id()
                self.tracks[tid] = {
                    "id": tid, "label": label, "xyxy": xyxy, "aabb": aabb,
                    "center": [cx,cy,cz], "seen": 1, "last": float(stamp),
                    "color_vals": o.get("color_vals", [[-1,-1,-1]]*3),
                    "color_labels": o.get("color_labels", ["N/A"]*3),
                    "color_percentages": [float(x) for x in o.get("color_percentages",[0.0,0.0,0.0])],
                }
            else:
                # 같은 물체로 판명되면 정보를 융합하지 않고,
                # seen 카운트와 마지막 관측 시간만 갱신합니다.
                tk = self.tracks[best_id]
                tk["seen"] += 1
                tk["last"] = float(stamp)

    def export_merged_simple(self, frame: str, params: SceneGraphParams) -> Dict[str,Any]:
        objects=[]
        for tid, tk in self.tracks.items():
            mn=tk["aabb"]["min"]; mx=tk["aabb"]["max"]
            objects.append({
                "id": int(tid),
                "label": tk["label"],
                "center": [float(tk["center"][0]), float(tk["center"][1]), float(tk["center"][2])],
                "aabb": {"min":[float(mn[0]),float(mn[1]),float(mn[2])],
                         "max":[float(mx[0]),float(mx[1]),float(mx[2])]},
                "footprint": [float(tk["xyxy"][0]), float(tk["xyxy"][1]), float(tk["xyxy"][2]), float(tk["xyxy"][3])],
                "npoints": 0,
            })
        if not objects:
            return {"frame": frame, "timestamp": float(rospy.Time.now().to_sec()), "objects": [], "relations": {}}
        sg = build_scene_graph(objects=objects, frame=frame, robot_xyyaw=[0.0,0.0,0.0], params=params)
        sg["timestamp"] = float(rospy.Time.now().to_sec())
        return sg

    def export_merged_rich(self, scene_name: str, region_name: str) -> Dict[str,Any]:
        objects=[]
        all_mins = np.array([+1e9, +1e9, +1e9], dtype=np.float32)
        all_maxs = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
        for tid, tk in self.tracks.items():
            mn=tk["aabb"]["min"]; mx=tk["aabb"]["max"]
            cx,cy,cz = tk["center"]
            objects.append({
                "id": int(tid),
                "label": tk["label"],
                "center": [float(cx), float(cy), float(cz)],
                "aabb": {"min":[float(mn[0]),float(mn[1]),float(mn[2])],
                         "max":[float(mx[0]),float(mx[1]),float(mx[2])]},
                "footprint": [float(tk["xyxy"][0]), float(tk["xyxy"][1]), float(tk["xyxy"][2]), float(tk["xyxy"][3])],
                "npoints": 0,
            })
            all_mins = np.minimum(all_mins, np.array(mn))
            all_maxs = np.maximum(all_maxs, np.array(mx))

        if not objects:
            return {
              "scene_name": scene_name,
              "regions": {"0": {"region_id":"0","region_name":region_name,
                                "region_bbox": _aabb_to_corners([0,0,0],[0,0,0]),
                                "objects":[], "relationships":{}}}
            }

        params = SceneGraphParams(
            near_dist=0.75,
            overlap_ratio=0.05,
            on_gap=0.25,
            above_gap=0.35,
            lr_thresh=0.05,
            fb_thresh=0.05,
            between_lat_thresh=0.5,
            max_between_pairs_per_anchor=2,
        )
        sg_simple = build_scene_graph(objects=objects, frame="map",
                                      robot_xyyaw=[0.0,0.0,0.0], params=params)

        margin = 0.1
        span = np.maximum(all_maxs - all_mins, margin)
        all_mins = all_mins - 0.05*span
        all_maxs = all_mins + 0.05*span
        region_bbox = _aabb_to_corners(all_mins.tolist(), all_maxs.tolist())

        rich_objs=[]
        for tid, tk in self.tracks.items():
            mn=tk["aabb"]["min"]; mx=tk["aabb"]["max"]
            size = (np.array(mx)-np.array(mn))
            vol = float(size[0]*size[1]*size[2])
            nyu = _nyu_lookup(tk["label"])
            rich_objs.append({
                "object_id": str(int(tid)),
                "raw_label": tk["label"],
                "nyu_id": nyu["nyu_id"],
                "nyu40_id": nyu["nyu40_id"],
                "nyu_label": nyu["nyu_label"],
                "nyu40_label": nyu["nyu40_label"],
                "color_vals": tk["color_vals"],
                "color_labels": tk["color_labels"],
                "color_percentages": [float(x) for x in tk["color_percentages"]],
                "bbox": _aabb_to_corners(mn, mx),
                "center": [float(tk["center"][0]), float(tk["center"][1]), float(tk["center"][2])],
                "volume": vol,
                "size": [float(size[0]), float(size[1]), float(size[2])],
                "affordances": []
            })

        return {
          "scene_name": scene_name,
          "regions": {
            "0": {
              "region_id": "0",
              "region_name": region_name,
              "region_bbox": region_bbox,
              "objects": rich_objs,
              "relationships": sg_simple.get("relations", {})
            }
          }
        }

# ----------------- ROS Node -----------------
class OneShotFusionNode:
    def __init__(self):
        rospy.init_node("fusion_pipeline_node", anonymous=False)
        self.bridge = CvBridge()
        # TF buffer 넉넉히
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(180.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Params
        self.min_points   = int(rospy.get_param("~min_points", 30))
        self.dump_dir     = rospy.get_param("~dump_dir", "/tmp/fusion_faas")
        os.makedirs(self.dump_dir, exist_ok=True)

        # 세션 루트
        self.session_dir = os.path.join(self.dump_dir, now_ts_str())
        os.makedirs(self.session_dir, exist_ok=True)

        self._run_count = 0
        self.map_frame    = rospy.get_param("~map_frame", "odom")
        self.prompt_topic = rospy.get_param("~prompt_topic", "/instruction")
        # self.seg_prompt   = rospy.get_param("~seg_prompt", "Segment salient, traversability-relevant objects (up to strictly 5).")
        self.seg_prompt = rospy.get_param("~seg_prompt", "")
        self.scene_name   = rospy.get_param("~scene_name", rospy.get_param("/current_scene_name", "unknown_scene"))
        self.region_name  = rospy.get_param("~region_name", "default_region")

        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.publish_goal_on_single = bool(rospy.get_param("~publish_goal_on_single", False))

        self.use_waypoint_interface    = bool(rospy.get_param("~use_waypoint_interface", True))
        self.waypoint_frame            = rospy.get_param("~waypoint_frame", "odom")
        self.waypoint_publish_rate     = float(rospy.get_param("~waypoint_publish_rate", 10.0))
        self.waypoint_publish_duration = float(rospy.get_param("~waypoint_publish_duration", 6.0))
        self.waypoint_only_if_single   = bool(rospy.get_param("~waypoint_only_if_single", True))

        self.sg_params  = SceneGraphParams(
            near_dist            = float(rospy.get_param("~sg_near_dist", 0.8)),
            overlap_ratio        = float(rospy.get_param("~sg_overlap_ratio", 0.2)),
            on_gap               = float(rospy.get_param("~sg_on_gap", 0.15)),
            above_gap            = float(rospy.get_param("~sg_above_gap", 0.20)),
            lr_thresh            = float(rospy.get_param("~sg_lr_thresh", 0.20)),
            fb_thresh            = float(rospy.get_param("~sg_fb_thresh", 0.20)),
            between_lat_thresh   = float(rospy.get_param("~sg_between_lat_thresh", 0.30)),
            max_between_pairs_per_anchor = int(rospy.get_param("~sg_between_max_pairs", 2)),
        )
        self.sg_enable  = bool(rospy.get_param("~sg_enable", True))

        # 자동 발행(정지 감지)
        self.auto_on_stop       = bool(rospy.get_param("~auto_on_stop", True))
        self.odom_topic         = rospy.get_param("~odom_topic", "/odom")
        self.stop_lin_thresh    = float(rospy.get_param("~stop_lin_thresh", 0.02))
        self.stop_ang_thresh    = float(rospy.get_param("~stop_ang_thresh", 0.02))
        self.stop_dwell_sec     = float(rospy.get_param("~stop_dwell_sec", 1.0))
        self.auto_cooldown_sec  = float(rospy.get_param("~auto_cooldown_sec", 8.0))

        # 세그 타임아웃 + 워치독
        self.seg_timeout_sec    = float(rospy.get_param("~seg_timeout_sec", 20.0))
        self._last_heartbeat    = time.time()
        self._stage             = "idle"

        self.merger = SGMerger(
            iou_thresh=float(rospy.get_param("~merge_iou_thresh", 0.25)),
            center_dist=float(rospy.get_param("~merge_center_dist", 0.75))
        )

        model_name = rospy.get_param("~gemini_model", "gemini-2.5-flash")
        self.segmenter = GeminiSegmenter(model=model_name)

        cam = DEFAULT_CAMERA.copy()
        cam.update(rospy.get_param("~camera", {}))
        self.CAM = {
            "width":  int(cam["width"]),
            "height": int(cam["height"]),
            "hfov_deg": float(cam["hfov_deg"]),
            "vfov_deg": float(cam["vfov_deg"]),
            "x": rospy.get_param("~l2c/x", DEFAULT_L2C["x"]),
            "y": rospy.get_param("~l2c/y", DEFAULT_L2C["y"]),
            "z": rospy.get_param("~l2c/z", DEFAULT_L2C["z"]),
            "roll":  rospy.get_param("~l2c/roll",  DEFAULT_L2C["roll"]),
            "pitch": rospy.get_param("~l2c/pitch", DEFAULT_L2C["pitch"]),
            "yaw":   rospy.get_param("~l2c/yaw",   DEFAULT_L2C["yaw"]),
        }
        self.LIDAR = {
            "x": rospy.get_param("~lidar/x"),
            "y": rospy.get_param("~lidar/y"),
            "z": rospy.get_param("~lidar/z"),
            "roll":  rospy.get_param("~lidar/roll"),
            "pitch": rospy.get_param("~lidar/pitch"),
            "yaw":   rospy.get_param("~lidar/yaw"),
        }

        self.lock = threading.Lock()
        self.is_busy = False
        self.last_image: Optional[PILImage.Image] = None
        self.last_image_np: Optional[np.ndarray] = None
        self.last_cloud_np: Optional[np.ndarray] = None
        self.last_cloud_stamp: Optional[rospy.Time] = None
        self.last_image_stamp: Optional[rospy.Time] = None
        self.last_cloud_frame: Optional[str] = None

        self.pub_centroids_map = rospy.Publisher("/fusion/centroids_map", PoseArray, queue_size=1, latch=True)
        self.pub_status        = rospy.Publisher("/fusion/status", String, queue_size=1)
        self.pub_segments      = rospy.Publisher("/fusion/segments", String, queue_size=1)
        self.pub_markers       = rospy.Publisher("/fusion/markers", MarkerArray, queue_size=1, latch=True)
        self.pub_goal          = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        self.pub_wp            = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=10)
        self.pub_resume        = rospy.Publisher("/resume_navigation", Empty, queue_size=1, latch=True)
        self.pub_resume_goal   = rospy.Publisher("/resume_navigation_to_goal", Empty, queue_size=1, latch=True)

        self.pub_scene_graph              = rospy.Publisher("/fusion/scene_graph", String, queue_size=1, latch=True)
        self.pub_scene_graph_rich         = rospy.Publisher("/fusion/scene_graph_rich", String, queue_size=1, latch=True)
        # LLM이 기대하는 스키마로 변환된 병합 SG는 기존 토픽(/fusion/scene_graph_merged)에 퍼블리시
        # 원본(simple) 병합 SG는 새 토픽(/fusion/scene_graph_merged_raw)에 함께 퍼블리시
        self.pub_scene_graph_merged       = rospy.Publisher("/fusion/scene_graph_merged", String, queue_size=1, latch=True)
        # self.pub_scene_graph_merged_raw   = rospy.Publisher("/fusion/scene_graph_merged_raw", String, queue_size=1, latch=True)
        self.pub_scene_graph_rich_merged  = rospy.Publisher("/fusion/scene_graph_rich_merged", String, queue_size=1, latch=True)
        # self.pub_scene_graph_image        = rospy.Publisher("/image_scenegraph", Image, queue_size=1)  # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
        # gpt: rel 이미지 저장 경로 준비 (/tmp/rel)
        # gpt: 설명.. /image_scenegraph 토픽을 쓰지 않고 파일로 저장하여 후단(gpt_llm)에서 직접 스캔하도록 변경
        self.rel_dir = "/tmp/rel"
        try:
            os.makedirs(self.rel_dir, exist_ok=True)
        except Exception as e:
            rospy.logwarn(f"[Fusion] rel dir create failed: {e}")

        self.image_topic = rospy.get_param("~camera_image_topic", "/camera/image")
        self.cloud_topic_a = rospy.get_param("~cloud_topic_a", "/color_scan_rel")
        self.cloud_topic_b = rospy.get_param("~cloud_topic_b", "/color_scan_relative")

        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.cloud_topic_a, PointCloud2, self.cb_cloud, queue_size=1)
        rospy.Subscriber(self.cloud_topic_b, PointCloud2, self.cb_cloud, queue_size=1)
        rospy.Subscriber(self.prompt_topic, String, self.cb_prompt, queue_size=10)
        rospy.Service("/fusion/run_once", Trigger, self.srv_run_once)
        rospy.Subscriber("/exploration_done", Bool, self.cb_exploration_done, queue_size=1)
        # 정지 감지 (오도메트리)
        self._stop_since = None
        self._auto_last_fire = 0.0
        if self.auto_on_stop:
            rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=20)

        # TF 기반 정지감지 폴백
        self.tf_stop_enable = bool(rospy.get_param("~tf_stop_enable", True))
        self.tf_stop_parent = rospy.get_param("~tf_stop_parent", self.map_frame)
        self.tf_stop_child  = rospy.get_param("~tf_stop_child",  "base_link")
        self._tf_stop_last  = None
        if self.auto_on_stop and self.tf_stop_enable:
            self._tf_timer = rospy.Timer(rospy.Duration(0.1), self._cb_tf_stop_poll)

        # 워치독
        self._watchdog = rospy.Timer(rospy.Duration(1.0), self._cb_watchdog)

        self.cooldown_sec = float(rospy.get_param("~cooldown_sec", 10.0))
        self.last_prompt_time = 0.0

        #image와 lidar 보관용 deque : 멈추면 push, segmentation할 때 pop
        #최대 보관 개수로 최대한 안 밀리는 정도에 따라 조절이 필요할 수도
        self.stack_maxlen=10
        self.item_stack=deque(maxlen=self.stack_maxlen)
        self.popped_item=None
        self._cap_seq=0 

        rospy.loginfo("fusion_pipeline_node ready. (one-shot + auto-on-stop)")

    # -------------------- SG 스키마 변환 유틸 (LLM 친화)
    def _relations_dict_to_list(self, rels: Dict[str, Any]) -> List[List[Any]]:
        """relations dict -> [[src, predicate, dst], ...]"""
        out = []
        try:
            if isinstance(rels, dict):
                for pred, pairs in rels.items():
                    try:
                        for p in (pairs or []):
                            # p가 [src, dst] 혹은 dict 형식일 수 있음
                            if isinstance(p, (list, tuple)) and len(p) >= 2:
                                out.append([int(p[0]), str(pred), int(p[1])])
                            elif isinstance(p, dict) and "src" in p and "dst" in p:
                                out.append([int(p["src"]), str(pred), int(p["dst"])])
                    except Exception as e:
                        rospy.loginfo(f"[Fusion] : skip relation predicate={pred} err={e}")
        except Exception as e:
            rospy.loginfo(f"[Fusion] : relations convert error: {e}")
        return out

    # def _simple_to_llm_partial(self, merged_simple: Dict[str, Any]) -> Dict[str, Any]:
    #     """simple merged SG(dict) -> LLM이 기대하는 최소 스키마로 변환"""
    #     objs_in = merged_simple.get("objects", []) if isinstance(merged_simple, dict) else []
    #     rels_in = merged_simple.get("relations", {}) if isinstance(merged_simple, dict) else {}
    #     objs = []
    #     for o in objs_in:
    #         try:
    #             oid = o.get("id")
    #             lbl = o.get("label")
    #             ctr = o.get("center")
    #             objs.append({
    #                 "object_id": int(oid) if oid is not None else None,
    #                 "raw_label": str(lbl) if lbl is not None else None,
    #                 "center": ctr,
    #             })
    #         except Exception as e:
    #             rospy.loginfo(f"[Fusion] : skip object in llm-partial err={e}")
    #     rels = self._relations_dict_to_list(rels_in)
    #     llm = {
    #         "scene_name": self.scene_name,
    #         "objects": objs,
    #         "relationships": rels,
    #     }
    #     return llm

    # -------------------- 공용 유틸
    def cb_exploration_done(self,msg : Bool):
        rospy.signal_shutdown("")
    def _lookup_tf_latest(self, target: str, source: str, stamp: rospy.Time):
        """요청 시각 lookup 실패 시 최신 시각(0)으로 재시도"""
        try:
            return self.tf_buffer.lookup_transform(target, source, stamp, rospy.Duration(0.5))
        except Exception:
            return self.tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(0.5))

    def _beat(self, stage: str):
        self._stage = stage
        self._last_heartbeat = time.time()

    # -------------------- 워치독
    def _cb_watchdog(self, _evt):
        if not self.is_busy:
            return
        now = time.time()
        if now - self._last_heartbeat > (self.seg_timeout_sec + 5.0):
            rospy.logwarn(f"[fusion] watchdog: stage='{self._stage}' stalled. Forcing reset.")
            with self.lock:
                self.is_busy = False
            self._publish_status(f"watchdog_reset stage={self._stage}")

    # -------------------- 세그 타임아웃 래퍼
    def _seg_with_timeout(self, image, prompt, timeout_sec: float):
        result = {}
        error = {}

        def _runner():
            try:
                items, raw, rel = self.segmenter.run_once(image, prompt=prompt)
                result["items"] = items
                result["raw"] = raw
                result["rel_score"] = rel if rel is not None else self.segmenter.last_rel_score
            except Exception as e:
                error["e"] = e

        th = threading.Thread(target=_runner, daemon=True)
        th.start()
        th.join(timeout_sec)

        if th.is_alive():
            raise TimeoutError(f"segmentation timeout after {timeout_sec:.1f}s")
        if "e" in error:
            raise error["e"]
        return result["items"], result["raw"], result.get("rel_score")

    def push_items(self):
        try:
            cap_id=self._cap_seq
            self.item_stack.append({
                "id": cap_id,
                "image_pil": self.last_image.copy(),
                "image_np":self.last_image_np.copy(),
                "cloud_np": self.last_cloud_np.copy(),
                "cloud_stamp": self.last_cloud_stamp,
                "image_stamp": self.last_image_stamp,
                "cloud_frame":self.last_cloud_frame
            })
            self._cap_seq+=1
        except Exception as e:
            rospy.logwarn(f"[fusion] {e}")
    # -------------------- 자동 발행(정지 감지) 공통 로직
    def _auto_stop_logic(self, lin, ang):
        now = time.time()
        stopped = (lin <= self.stop_lin_thresh) and (ang <= self.stop_ang_thresh)
        if stopped:
            if self._stop_since is None:
                self._stop_since = now
            dwell = now - self._stop_since
            if dwell >= self.stop_dwell_sec:
                if (now - self._auto_last_fire) >= self.auto_cooldown_sec:
                    #이미지와 lidar의 발행 hz 고려해서 멈췄을 때의 이미지와 lidar가 발행이 된 후가 보장이 되도록 sleep : 센서 5hz 이미지 10hz
                    time.sleep(0.5)
                    with self.lock:
                        if (not self.is_busy) and (self.last_image is not None) and (self.last_cloud_np is not None):
                            self.is_busy = True
                            self.push_items()
                            # published_image=CvBridge().cv2_to_imgmsg(self.last_image_np, encoding="rgb8")
                            # rospy.loginfo("[Fusion] : publishing /image_scenegraph just before pipeline run")
                            # self.pub_scene_graph_image.publish(published_image)
                            # 원래 코드들 ... [image_scenegraph, rel_score 삭제에 따른 주석 처리]
                            self._auto_last_fire = now
                            rospy.loginfo(f"[auto] stopped for {dwell:.2f}s -> run pipeline once")
                            threading.Thread(
                                target=self._run_once_pipeline,
                                args=(self.seg_prompt or "",),
                                daemon=True
                            ).start()
        else:
            self._stop_since = None

    def _cb_tf_stop_poll(self, _evt):
        if not self.tf_stop_enable:
            return
        try:
            tfm = self._lookup_tf_latest(self.tf_stop_parent, self.tf_stop_child, rospy.Time(0))
        except Exception:
            return
        now = time.time()
        p = tfm.transform.translation
        q = tfm.transform.rotation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        cur = (now, float(p.x), float(p.y), float(yaw))
        if self._tf_stop_last is None:
            self._tf_stop_last = cur
            return
        t0, x0, y0, yaw0 = self._tf_stop_last
        dt = max(1e-3, cur[0] - t0)
        lin = math.hypot(cur[1] - x0, cur[2] - y0) / dt
        dyaw = (cur[3] - yaw0 + math.pi) % (2*math.pi) - math.pi
        ang = abs(dyaw) / dt
        self._tf_stop_last = cur
        self._auto_stop_logic(lin, ang)

    # -------------------- 자동 발행(정지 감지) - 오도메트리
    def cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)
        lin = math.hypot(vx, vy)
        ang = abs(wz)
        self._auto_stop_logic(lin, ang)

    # -------------------- 수동 트리거 (busy → cooldown 순)
    def cb_prompt(self, msg: String):
        with self.lock:
            if self.is_busy:
                rospy.logwarn("[fusion] busy; ignoring /instruction")
                return
        if time.time() - self.last_prompt_time < self.cooldown_sec:
            rospy.logwarn("[fusion] prompt cooldown; ignored")
            return
        self.last_prompt_time = time.time()
        with self.lock:
            if self.last_image is None or self.last_cloud_np is None:
                rospy.logwarn("[fusion] no cached image/cloud; ignoring /instruction")
                return
            self.is_busy = True
        threading.Thread(
            target=self._run_once_pipeline,
            args=(msg.data or self.seg_prompt or "",),
            daemon=True
        ).start()

    # -------------------- 콜백들
    def cb_image(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv_bgr[..., ::-1].copy()
            self.last_image_np = rgb
            self.last_image = PILImage.fromarray(rgb)
            self.last_image_stamp = msg.header.stamp
        except Exception as e:
            rospy.logwarn(f"[image] cv_bridge failed: {e}")

    def cb_cloud(self, msg: PointCloud2):
        arr, frame_id, stamp = cloud2_to_xyz_array(msg)
        self.last_cloud_np = arr
        self.last_cloud_stamp = stamp
        self.last_cloud_frame = frame_id or "base_link"

    def srv_run_once(self, req: TriggerRequest) -> TriggerResponse:
        with self.lock:
            if self.is_busy:
                return TriggerResponse(success=False, message="Busy")
            if self.last_image is None or self.last_cloud_np is None:
                return TriggerResponse(success=False, message="No image/cloud cached yet")
            self.is_busy = True
        threading.Thread(target=self._run_once_pipeline, args=(self.seg_prompt or "",), daemon=True).start()
        return TriggerResponse(success=True, message="Started")

    # -------------------- Core pipeline
    def save_ply_xyz(self,path: str, xyz: np.ndarray):
        """
        xyz: (N,3) float 배열. NaN/Inf은 자동 제거.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = np.asarray(xyz, dtype=np.float32)
        # NaN/Inf 제거
        mask = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[mask]
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    def _run_once_pipeline(self, prompt: str):
        with self.lock:
            self._run_count += 1
            run_idx = self._run_count
        out_root = self.session_dir
        seg_dir = os.path.join(out_root, "seg");   os.makedirs(seg_dir, exist_ok=True)
        dbg_dir = os.path.join(out_root, "debug"); os.makedirs(dbg_dir, exist_ok=True)
        rel_session_dir = os.path.join(out_root, "rel"); os.makedirs(rel_session_dir, exist_ok=True)

        try:
            self.popped_item=self.item_stack.popleft()
            rospy.loginfo(f"stamp difference: {self.popped_item['image_stamp'] - self.popped_item['cloud_stamp']}| cloud stamp: {self.popped_item['cloud_stamp']}, image stamp: {self.popped_item['image_stamp']}")
            self._beat("start")
            self._dump_inputs(out_root)
            self._beat("dumped")

            # 1) segmentation
            try:
                self._beat("seg:start")
                eff_prompt = (prompt.strip() or self.seg_prompt.strip()) if (prompt or self.seg_prompt) else None
                items, raw_text, rel_score = self._seg_with_timeout(
                    self.popped_item["image_pil"],
                    prompt=eff_prompt,
                    timeout_sec=self.seg_timeout_sec
                )
                # gpt: 설명.. rel_score는 API에서 계산되며 topic publish는 사용하지 않음. 이후 파일 저장에 사용.
                self._beat(f"seg:done({len(items)})")
            except Exception as e:
                self._publish_status(f"Seg error: {e}")
                self._write_summary(out_root, "seg_error", [], [], [], [], self.popped_item["cloud_frame"] or "")
                return

            self._save_seg_artifacts(seg_dir, items, raw_text)
            # gpt: 설명.. rel_score 기반 파일 저장 (/tmp/rel/{ts}_{relscore}.png)
            try:
                ts = now_ts_str()
                score_str = "unknown"
                if isinstance(rel_score, int) and 0 <= rel_score <= 100:
                    score_str = str(rel_score)
                else:
                    rospy.logwarn(f"[Fusion] rel_score invalid or missing; saving with 'unknown'")
                out_png = os.path.join(self.rel_dir, f"{ts}_{score_str}.png")
                rel_session_png = os.path.join(rel_session_dir, f"rel_{ts}_{score_str}.png")
                try:
                    self.popped_item["image_pil"].save(out_png, "PNG")
                    self.popped_item["image_pil"].save(rel_session_png, "PNG")
                    rospy.loginfo(f"[Fusion] saved rel image: {out_png}")
                except Exception as e:
                    rospy.logwarn(f"[Fusion] rel image save failed: {e}")
            except Exception as e:
                rospy.logwarn(f"[Fusion] rel image pipeline save failed: {e}")
            if not items:
                self._publish_status("Seg returned 0 items")
                self._write_summary(out_root, "seg_empty", [], [], [], [], self.popped_item["cloud_frame"] or "")
                return

            # 2) projection
            cloud_rel = self.popped_item["cloud_np"]
            if cloud_rel is None or cloud_rel.shape[0] == 0:
                self._publish_status("Empty cloud")
                self._write_summary(out_root, "empty_cloud", [], [], [], [], self.popped_item["cloud_frame"] or "")
                return

            uvd = scan2pixels_mecanum(cloud_rel, self.CAM, self.LIDAR)
            u = uvd[:, 0].astype(np.intp, copy=False)
            v = uvd[:, 1].astype(np.intp, copy=False)
            draw_points_overlay(self.popped_item["image_pil"], u, v, None, os.path.join(dbg_dir, f"points_on_image_{now_ts_str()}.png"))
            H, W = int(self.CAM["height"]), int(self.CAM["width"])

            def _bbox_from_box2d(b):
                if not (isinstance(b, list) and len(b) == 4): return None
                y0a = int(b[0]/1000.0*H); x0a = int(b[1]/1000.0*W)
                y1a = int(b[2]/1000.0*H); x1a = int(b[3]/1000.0*W)
                x0b = int(b[0]/1000.0*W); y0b = int(b[1]/1000.0*H)
                x1b = int(b[2]/1000.0*W); y1b = int(b[3]/1000.0*H)
                def ok(x0,y0,x1,y1): return (x1>x0) and (y1>y0)
                if ok(x0a,y0a,x1a,y1a): return x0a,y0a,x1a,y1a
                if ok(x0b,y0b,x1b,y1b): return x0b,y0b,x1b,y1b
                return None

            try:
                any_pick = np.zeros(u.shape[0], dtype=bool)
                for it in items:
                    bb = _bbox_from_box2d(it.box_2d)
                    if bb is None: continue
                    x0,y0,x1,y1 = bb
                    any_pick |= ((u >= x0) & (u <= x1) & (v >= y0) & (v <= y1))
                draw_points_overlay(self.popped_item["image_pil"], u, v, any_pick, os.path.join(dbg_dir, "points_on_image.png"))
            except Exception as e:
                rospy.logwarn(f"[debug] overlay-all failed: {e}")

            centroids_rel, labels, picks_per_seg = [], [], []
            segments_out = []
            obj_pts_rel_list = []
            rgb_img = self.popped_item["image_np"]

            for idx, it in enumerate(items):
                it.label = _safe_label(getattr(it, "label", "unknown"))
                bb = _bbox_from_box2d(it.box_2d)
                if bb is None:
                    picks_per_seg.append(0)
                    segments_out.append({
                        "index": idx, "label": it.label, "box_px": None,
                        "box_px_expanded": None, "picks": 0,
                        "centroid_rel": None, "centroid_map": None,
                        "color_vals":[[-1,-1,-1]]*3,"color_labels":["N/A"]*3,"color_percentages":[0.0,0.0,0.0],
                    })
                    obj_pts_rel_list.append(None)
                    continue

                x0, y0, x1, y1 = bb
                x0 = max(0, min(W-1, x0)); x1 = max(0, min(W-1, x1))
                y0 = max(0, min(H-1, y0)); y1 = max(0, min(H-1, y1))

                cloud_mask = None
                full_mask_img=None
                try:
                    mbytes = decode_data_url_to_bytes(getattr(it, "mask", "") or "")
                    if mbytes:
                        mimg = PILImage.open(io.BytesIO(mbytes)).convert("L")
                        mimg = mimg.resize((max(1, x1-x0), max(1, y1-y0)), PILImage.Resampling.BILINEAR)
                        marr = np.array(mimg)
                        full = np.zeros((H, W), dtype=np.uint8)
                        full[y0:y1, x0:x1] = marr
                        full_mask_img = full.copy()
                        cloud_mask = (full[v, u] > 127)
                except Exception:
                    cloud_mask = None
                if full_mask_img is not None:
                    try:
                        base = (self.popped_item["image_pil"] or PILImage.fromarray(self.popped_item["image_np"])).copy().convert("RGBA")
                        Wb, Hb = base.size
                        if (Wb, Hb) != (W, H):
                            full_for_overlay = PILImage.fromarray(full_mask_img, mode="L").resize(
                                (Wb, Hb), PILImage.Resampling.NEAREST
                            )
                            full_arr = np.array(full_for_overlay)
                        else:
                            full_arr = full_mask_img
                        overlay_rgba = np.zeros((Hb, Wb, 4), dtype=np.uint8)
                        overlay_rgba[full_arr > 127] = (0, 255, 0, 120)
                        over_img = PILImage.fromarray(overlay_rgba, mode="RGBA")
                        merged = PILImage.alpha_composite(base, over_img)

                        safe_label = re.sub('[^A-Za-z0-9_-]', '_', it.label)[:40]
                        out_path = os.path.join(dbg_dir, f"seg_mask_proj_{idx}_{safe_label}_{now_ts_str()}.png")
                        merged.save(out_path)
                    except Exception as e:
                        rospy.logwarn(f"[debug] mask-overlay seg#{idx} failed: {e}")
                if cloud_mask is None:
                    cloud_mask = ((u >= x0) & (u <= x1) & (v >= y0) & (v <= y1))

                idxs = np.where(cloud_mask)[0]

                xa = xb = ya = yb = None
                if idxs.size < self.min_points:
                    dx = int(0.1 * (x1-x0+1)); dy = int(0.1 * (y1-y0+1))
                    xa = max(0, x0-dx); ya = max(0, y0-dy)
                    xb = min(W-1, x1+dx); yb = min(H-1, y1+dy)
                    idxs = np.where((u >= xa) & (u <= xb) & (v >= ya) & (v <= yb))[0]

                if xa is not None:
                    rospy.loginfo(f"[fusion] seg#{idx} '{it.label}' picks={idxs.size} box=[{x0},{y0},{x1},{y1}] expanded=[{xa},{ya},{xb},{yb}]")
                else:
                    rospy.loginfo(f"[fusion] seg#{idx} '{it.label}' picks={idxs.size} box=[{x0},{y0},{x1},{y1}]")

                picks_per_seg.append(int(idxs.size))

                # 색상 분석
                c_vals=[[-1,-1,-1]]*3; c_labels=["N/A"]*3; c_percs=[0.0,0.0,0.0]
                if idxs.size >= self.min_points and rgb_img is not None:
                    try:
                        xs = np.clip(u[idxs], 0, W-1).astype(np.intp, copy=False)
                        ys = np.clip(v[idxs], 0, H-1).astype(np.intp, copy=False)
                        pix = rgb_img[ys, xs, :]
                        c_vals, c_labels, c_percs, _ = _top3_color_stats(pix)
                    except Exception as e:
                        rospy.logwarn(f"[color] sampling failed on seg#{idx}: {e}")

                try:
                    pick_mask = np.zeros(u.shape[0], dtype=bool); pick_mask[idxs] = True
                    safe_label = re.sub('[^A-Za-z0-9_-]', '_', it.label)[:40]
                    draw_points_overlay(self.popped_item["image_pil"], u, v, pick_mask,
                        os.path.join(dbg_dir, f"seg_pick_{idx}_{safe_label}.png"),
                        max_points=60000)
                except Exception as e:
                    rospy.logwarn(f"[debug] overlay seg#{idx} failed: {e}")

                seg_info = {
                    "index": idx,
                    "label": it.label,
                    "box_px": [int(x0), int(y0), int(x1), int(y1)],
                    "box_px_expanded": ([int(xa), int(ya), int(xb), int(yb)] if xa is not None else None),
                    "picks": int(idxs.size),
                    "centroid_rel": None,
                    "centroid_map": None,
                    "color_vals": c_vals,
                    "color_labels": c_labels,
                    "color_percentages": c_percs,
                }

                if idxs.size < self.min_points:
                    segments_out.append(seg_info)
                    obj_pts_rel_list.append(None)
                    continue

                pts = cloud_rel[idxs, :]
                """
                fname_xyz = os.path.join(debug_dir, f"seg_{idx:02d}_{safe_label}_rel_{now_ts_str()}.ply")
                self.save_ply_xyz(fname_xyz, pts)
                """
                try:
                    c_rel = np.median(pts, axis=0)
                except Exception:
                    c_rel = pts.mean(axis=0)

                mins = pts.min(axis=0); maxs = pts.max(axis=0)
                seg_info["stats_rel"] = {
                    "aabb_min": [float(mins[0]), float(mins[1]), float(mins[2])],
                    "aabb_max": [float(maxs[0]), float(maxs[1]), float(maxs[2])],
                    "footprint": [float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])],
                    "npoints": int(pts.shape[0]),
                }

                c_rel_list = c_rel.astype(float).tolist()
                centroids_rel.append(c_rel_list)
                labels.append(it.label or f"segment_{idx}")
                seg_info["centroid_rel"] = c_rel_list
                segments_out.append(seg_info)
                obj_pts_rel_list.append(pts)

            if len(centroids_rel) == 0:
                self._publish_status("No centroids (too few points per segment)")
                self._write_summary(out_root, "no_centroids", labels, picks_per_seg, [], [], self.popped_item["cloud_frame"] or "")
                self._publish_segments(segments_out)
                return

            # 4) TF rel->map
            frame_src = self.popped_item["cloud_frame"] or "base_link"
            stamp = self.popped_item["cloud_stamp"] or rospy.Time(0)
            try:
                self._beat("tf:start")
                tfm = self._lookup_tf_latest(self.map_frame, frame_src, stamp)
                t = tfm.transform.translation
                q = tfm.transform.rotation
                R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
                tvec = np.array([t.x, t.y, t.z], dtype=np.float32)
            except Exception as e:
                self._publish_status(f"TF lookup failed: {self.map_frame}<-{frame_src}: {e}")
                self._write_summary(out_root, "tf_fail", labels, picks_per_seg, centroids_rel, [], frame_src)
                self._publish_segments(segments_out)
                return

            centroids_map = []
            for c in centroids_rel:
                c_rel = np.array(c, dtype=np.float32)
                c_map = R @ c_rel + tvec
                centroids_map.append(c_map.astype(float).tolist())

            it_cm = iter(centroids_map)
            for seg in segments_out:
                if seg["centroid_rel"] is not None:
                    seg["centroid_map"] = next(it_cm, None)

            self._beat("tf:done")

            # 5) PoseArray
            pa = PoseArray()
            pa.header.stamp = rospy.Time.now()
            pa.header.frame_id = self.map_frame
            for cm in centroids_map:
                pose = Pose()
                pose.position = Point(*cm)
                pose.orientation = Quaternion(0,0,0,1)
                pa.poses.append(pose)
            self.pub_centroids_map.publish(pa)

            # 6) Scene Graph (스냅샷)
            try:
                self._beat("sg:start")
                objects_for_sg = []
                obj_id_running = 0
                rospy.loginfo(f"[Fusion] : preparing snapshot SG from {len(segments_out)} segments")
                for k, seg in enumerate(segments_out):
                    if seg.get("centroid_rel") is None: continue
                    pts_rel = obj_pts_rel_list[k]
                    if pts_rel is None or pts_rel.shape[0] == 0: continue
                    pts_map = (R @ pts_rel.T).T + tvec
                    mins = pts_map.min(axis=0); maxs = pts_map.max(axis=0)
                    footprint = [float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])]
                    cm = seg.get("centroid_map") or [float((mins[0]+maxs[0])/2), float((mins[1]+maxs[1])/2), float((mins[2]+maxs[2])/2)]
                    objects_for_sg.append({
                        "id": int(obj_id_running),
                        "label": str(seg.get("label","")),
                        "center": [float(cm[0]), float(cm[1]), float(cm[2])],
                        "aabb": {"min":[float(mins[0]), float(mins[1]), float(mins[2])],
                                 "max":[float(maxs[0]), float(maxs[1]), float(maxs[2])]},
                        "footprint": footprint,
                        "npoints": int(pts_rel.shape[0]),
                    })
                    seg["object_id"] = int(obj_id_running)
                    obj_id_running += 1

                scene_graph = None
                if self.sg_enable and len(objects_for_sg) > 0:
                    # robot pose (optional)
                    robot_xyyaw = [0.0, 0.0, 0.0]
                    try:
                        tfm_rb = self.tf_buffer.lookup_transform(self.map_frame, "base_link", rospy.Time(0), rospy.Duration(0.5))
                        rx = float(tfm_rb.transform.translation.x)
                        ry = float(tfm_rb.transform.translation.y)
                        rq = tfm_rb.transform.rotation
                        yaw = tft.euler_from_quaternion([rq.x, rq.y, rq.z, rq.w])[2]
                        robot_xyyaw = [rx, ry, float(yaw)]
                    except Exception:
                        pass

                    rospy.loginfo(f"[Fusion] : building snapshot SG with {len(objects_for_sg)} objects")
                    scene_graph = build_scene_graph(objects=objects_for_sg,
                                                    frame=self.map_frame,
                                                    robot_xyyaw=robot_xyyaw,
                                                    params=self.sg_params)
                    scene_graph["timestamp"] = float(rospy.Time.now().to_sec())
                    fname = f"scene_graph_{run_idx}.json"
                    with open(os.path.join(self.session_dir, fname), "w", encoding="utf-8") as f:
                        json.dump(scene_graph, f, ensure_ascii=False, indent=2)
                    rospy.loginfo(f"[Fusion] : publishing /fusion/scene_graph (snapshot) objs={len(scene_graph.get('objects',[]))}")
                    self.pub_scene_graph.publish(String(data=json.dumps(scene_graph)))
                else:
                    rospy.loginfo(f"[Fusion] : snapshot SG skipped (sg_enable={self.sg_enable}, objs={len(objects_for_sg)})")
                self._beat("sg:done")
            except Exception as e:
                rospy.logwarn(f"[scene-graph] build error: {e}")
                scene_graph = None

            # 7) move_base goal (옵션)
            try:
                if bool(self.publish_goal_on_single) and len(centroids_map) == 1:
                    gx, gy, gz = centroids_map[0]
                    yaw = 0.0
                    try:
                        tfm_rb = self.tf_buffer.lookup_transform(self.map_frame, "base_link", rospy.Time(0), rospy.Duration(0.5))
                        rx = float(tfm_rb.transform.translation.x)
                        ry = float(tfm_rb.transform.translation.y)
                        yaw = math.atan2(gy - ry, gx - rx)
                    except Exception:
                        pass
                    qz = math.sin(yaw / 2.0)
                    qw = math.cos(yaw / 2.0)
                    goal = PoseStamped()
                    goal.header.stamp = rospy.Time.now()
                    goal.header.frame_id = self.map_frame
                    goal.pose.position.x = float(gx)
                    goal.pose.position.y = float(gy)
                    goal.pose.position.z = 0.0
                    goal.pose.orientation.z = qz
                    goal.pose.orientation.w = qw
                    self.pub_goal.publish(goal)
                    self._publish_status(f"Sent nav goal to {self.goal_topic}: ({gx:.2f}, {gy:.2f})")
            except Exception as e:
                rospy.logwarn(f"[nav-goal] failed: {e}")

            # 8) waypoint 인터페이스
            try:
                if self.use_waypoint_interface and (not self.waypoint_only_if_single or len(centroids_map) == 1):
                    gx_map, gy_map, gz_map = centroids_map[0]
                    tx, ty = float(gx_map), float(gy_map)
                    if self.waypoint_frame and self.waypoint_frame != self.map_frame:
                        try:
                            tfm_w = self.tf_buffer.lookup_transform(self.waypoint_frame, self.map_frame, rospy.Time(0), rospy.Duration(0.5))
                            t = tfm_w.transform.translation
                            q = tfm_w.transform.rotation
                            Rmw = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
                            tw  = np.array([t.x, t.y, t.z], dtype=np.float32)
                            p_m = np.array([gx_map, gy_map, gz_map], dtype=np.float32)
                            p_w = Rmw @ p_m + tw
                            tx, ty = float(p_w[0]), float(p_w[1])
                        except Exception as e:
                            rospy.logwarn(f"[waypoint] TF {self.waypoint_frame}<-{self.map_frame} failed, use map coords: {e}")
                    yaw = 0.0
                    try:
                        tfm_rb = self.tf_buffer.lookup_transform(self.waypoint_frame, "base_link", rospy.Time(0), rospy.Duration(0.5))
                        rx = float(tfm_rb.transform.translation.x)
                        ry = float(tfm_rb.transform.translation.y)
                        yaw = math.atan2(ty - ry, tx - rx)
                    except Exception:
                        pass
                    self.pub_resume.publish(Empty())
                    self.pub_resume_goal.publish(Empty())
                    rate = rospy.Rate(self.waypoint_publish_rate)
                    t_end = rospy.Time.now() + rospy.Duration(self.waypoint_publish_duration)
                    msg = Pose2D(x=tx, y=ty, theta=yaw)
                    rospy.loginfo(f"[waypoint] publish {self.waypoint_publish_duration:.1f}s @ {self.waypoint_publish_rate:.1f}Hz => ({tx:.2f}, {ty:.2f}, yaw={yaw:.2f}) in {self.waypoint_frame}")
                    while not rospy.is_shutdown() and rospy.Time.now() < t_end:
                        self.pub_wp.publish(msg)
                        self.pub_resume.publish(Empty())
                        self.pub_resume_goal.publish(Empty())
                        rate.sleep()
                    self._publish_status(f"Sent waypoint Pose2D to /way_point_with_heading: ({tx:.2f}, {ty:.2f}) [{self.waypoint_frame}]")
            except Exception as e:
                rospy.logwarn(f"[waypoint] failed: {e}")

            # 9) RViz markers
            try:
                ma = MarkerArray()
                now = rospy.Time.now()
                ns = "fusion_segments"
                for i, (cm, lbl) in enumerate(zip(centroids_map, labels)):
                    m = Marker()
                    m.header.stamp = now
                    m.header.frame_id = self.map_frame
                    m.ns = ns
                    m.id = i*2
                    m.type = Marker.SPHERE
                    m.action = Marker.ADD
                    m.pose.position.x = float(cm[0])
                    m.pose.position.y = float(cm[1])
                    m.pose.position.z = float(cm[2])
                    m.pose.orientation.w = 1.0
                    m.scale.x = m.scale.y = m.scale.z = 0.25
                    m.color.a = 0.9; m.color.r = 1.0; m.color.g = 0.2; m.color.b = 0.2
                    m.lifetime = rospy.Duration(0)
                    ma.markers.append(m)

                    t = Marker()
                    t.header.stamp = now
                    t.header.frame_id = self.map_frame
                    t.ns = ns
                    t.id = i*2 + 1
                    t.type = Marker.TEXT_VIEW_FACING
                    t.action = Marker.ADD
                    t.pose.position.x = float(cm[0])
                    t.pose.position.y = float(cm[1])
                    t.pose.position.z = float(cm[2]) + 0.3
                    t.pose.orientation.w = 1.0
                    t.scale.z = 0.25
                    t.color.a = 1.0; t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
                    t.text = str(lbl)
                    t.lifetime = rospy.Duration(0)
                    ma.markers.append(t)
                self.pub_markers.publish(ma)
            except Exception as e:
                rospy.logwarn(f"[markers] publish failed: {e}")

            # 10) 요약 + 상태 + segments 퍼블리시
            self._write_summary(out_root, "ok", labels, picks_per_seg, centroids_rel, centroids_map, frame_src)
            status_payload = {"ok": True, "n": len(centroids_map), "labels": labels, "segments": segments_out}
            self._publish_status(json.dumps(status_payload))
            self._publish_segments(segments_out)

            # 11) RICH 생성/퍼블리시 + MERGE
            rich = None
            try:
                if scene_graph is not None:
                    all_mins = np.array([+1e9, +1e9, +1e9], dtype=np.float32)
                    all_maxs = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
                    for o in scene_graph.get("objects", []):
                        mn = np.array(o["aabb"]["min"]); mx = np.array(o["aabb"]["max"])
                        all_mins = np.minimum(all_mins, mn); all_maxs = np.maximum(all_maxs, mx)
                    margin = 0.1
                    span = np.maximum(all_maxs - all_mins, margin)
                    all_mins = all_mins - 0.05*span
                    all_maxs = all_mins + 0.05*span
                    region_bbox = _aabb_to_corners(all_mins.tolist(), all_maxs.tolist())

                    seg_by_oid = {}
                    for seg in segments_out:
                        oid = seg.get("object_id", None)
                        if oid is not None:
                            seg_by_oid[int(oid)] = seg

                    rich_objects=[]
                    for o in scene_graph.get("objects", []):
                        mn = np.array(o["aabb"]["min"]); mx = np.array(o["aabb"]["max"])
                        size = (mx - mn)
                        vol  = float(size[0]*size[1]*size[2])
                        nyu  = _nyu_lookup(o["label"])

                        seg_src = seg_by_oid.get(int(o["id"]))
                        if seg_src is not None:
                            c_vals, c_labels, c_percs, _ = _normalize_color_fields(seg_src)
                        else:
                            c_vals=[[-1,-1,-1]]*3; c_labels=["N/A"]*3; c_percs=[0.0,0.0,0.0]

                        rich_objects.append({
                            "object_id": str(o["id"]),
                            "raw_label": o["label"],
                            "nyu_id": nyu["nyu_id"],
                            "nyu40_id": nyu["nyu40_id"],
                            "nyu_label": nyu["nyu_label"],
                            "nyu40_label": nyu["nyu40_label"],
                            "color_vals": c_vals,
                            "color_labels": c_labels,
                            "color_percentages": c_percs,
                            "bbox": _aabb_to_corners(mn.tolist(), mx.tolist()),
                            "center": o["center"],
                            "volume": vol,
                            "size": [float(size[0]), float(size[1]), float(size[2])],
                            "affordances": []
                        })

                    rich_relationships = scene_graph.get("relations", {})
                    rich = {
                        "scene_name": self.scene_name,
                        "regions": {
                            "0": {
                                "region_id": "0",
                                "region_name": self.region_name,
                                "region_bbox": region_bbox,
                                "objects": rich_objects,
                                "relationships": rich_relationships
                            }
                        }
                    }

                    with open(os.path.join(out_root, "scene_graph_rich.json"), "w", encoding="utf-8") as f:
                        json.dump(rich, f, ensure_ascii=False, indent=2)
                    rospy.loginfo(f"[Fusion] : publishing /fusion/scene_graph_rich objs={len(rich.get('regions',{}).get('0',{}).get('objects',[]))}")
                    self.pub_scene_graph_rich.publish(String(data=json.dumps(rich)))
            except Exception as e:
                rospy.logwarn(f"[scene-graph rich] build/publish failed: {e}")

            try:
                if rich is not None:
                    ts_now = float(rospy.Time.now().to_sec())
                    self.merger.update_with_rich(rich, stamp=ts_now)
                    merged_simple = self.merger.export_merged_simple(self.map_frame, self.sg_params)
                    merged_rich   = self.merger.export_merged_rich(self.scene_name, self.region_name)
                    with open(os.path.join(self.session_dir, "scene_graph_merged.json"), "w", encoding="utf-8") as f:
                        json.dump(merged_simple, f, ensure_ascii=False, indent=2)
                    with open(os.path.join(self.session_dir, "scene_graph_rich_merged.json"), "w", encoding="utf-8") as f:
                        json.dump(merged_rich, f, ensure_ascii=False, indent=2)
                    # 원본(simple) 병합 SG는 raw 토픽으로 퍼블리시
                    rospy.loginfo(f"[Fusion] : publishing /fusion/scene_graph_merged objs={len(merged_simple.get('objects',[]))}")
                    self.pub_scene_graph_merged.publish(String(data=json.dumps(merged_simple)))

                    # LLM용 스키마로 변환하여 /fusion/scene_graph_merged 에 퍼블리시
                    # llm_merged = self._simple_to_llm_partial(merged_simple)
                    # rospy.loginfo(f"[Fusion] : publishing /fusion/scene_graph_merged (LLM) objs={len(llm_merged.get('objects',[]))} rels={len(llm_merged.get('relationships',[]))}")
                    # self.pub_scene_graph_merged.publish(String(data=json.dumps(llm_merged)))

                    rospy.loginfo(f"[Fusion] : publishing /fusion/scene_graph_rich_merged objs={len(merged_rich.get('regions',{}).get('0',{}).get('objects',[]))}")
                    self.pub_scene_graph_rich_merged.publish(String(data=json.dumps(merged_rich)))
            except Exception as e:
                rospy.logwarn(f"[scene-graph rich/merge] build/publish failed: {e}")

        except Exception as e:
            self._publish_status(f"Exception: {e}")
            self._write_summary(out_root, "exception", [], [], [], [], self.popped_item["cloud_frame"] or "")
        finally:
            with self.lock:
                self.is_busy = False
            self._beat("idle")

    # -------------------- dump / artifacts
    def _dump_inputs(self, out_root: str):
        os.makedirs(out_root, exist_ok=True)
        if self.popped_item["image_pil"] is not None:
            try:
                self.popped_item["image_pil"].save(os.path.join(out_root, "image.png"), "PNG")
            except Exception as e:
                rospy.logwarn(f"[dump] save image failed: {e}")
        if self.popped_item["cloud_np"] is not None and self.popped_item["cloud_np"].size > 0:
            try:
                write_pcd_float32(os.path.join(out_root, "cloud_rel.pcd"),
                                  self.popped_item["cloud_np"].astype(np.float32), ["x","y","z"])
            except Exception as e:
                rospy.logwarn(f"[dump] save rel cloud failed: {e}")
            frame_src = self.popped_item["cloud_frame"] or "base_link"
            stamp = self.popped_item["cloud_stamp"] or rospy.Time(0)
            try:
                tfm = self._lookup_tf_latest(self.map_frame, frame_src, stamp)
                t = tfm.transform.translation
                q = tfm.transform.rotation
                R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
                tvec = np.array([t.x, t.y, t.z], dtype=np.float32)
                pts_map = self.popped_item["cloud_np"].copy()
                pts_map[:, :3] = (R @ pts_map[:, :3].T).T + tvec
                write_pcd_float32(os.path.join(out_root, "cloud_map.pcd"),
                                  pts_map.astype(np.float32), ["x","y","z"])
            except Exception as e:
                rospy.logwarn(f"[dump] map transform failed: {e}")

    def _save_seg_artifacts(self, seg_dir: str, items: List[SegItem], raw_text: Optional[str]):
        os.makedirs(seg_dir, exist_ok=True)
        H, W = int(self.CAM["height"]), int(self.CAM["width"])
        out = []
        for it in items:
            b = it.box_2d or [0,0,0,0]
            y0a = int(b[0]/1000.0*H); x0a = int(b[1]/1000.0*W)
            y1a = int(b[2]/1000.0*H); x1a = int(b[3]/1000.0*W)
            x0b = int(b[0]/1000.0*W); y0b = int(b[1]/1000.0*H)
            x1b = int(b[2]/1000.0*W); y1b = int(b[3]/1000.0*H)
            out.append({"label": _safe_label(getattr(it, "label", "unknown")), "box_2d": b,
                        "box_px_tryA": [x0a,y0a,x1a,y1a], "box_px_tryB": [x0b,y0b,x1b,y1b],
                        "mask_len": len(getattr(it, "mask", "") or "")})
        with open(os.path.join(seg_dir, "items.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        try:
            base = self.popped_item["image_pil"].convert("RGBA")
            HH, WW = base.size[1], base.size[0]
            over = PILImage.new("RGBA", (WW,HH), (0,0,0,0))
            dr = ImageDraw.Draw(over)
            for i, it in enumerate(items):
                b = it.box_2d or [0,0,0,0]
                y0 = int(b[0]/1000.0*HH); x0 = int(b[1]/1000.0*WW)
                y1 = int(b[2]/1000.0*HH); x1 = int(b[3]/1000.0*WW)
                x0,x1 = sorted([x0,x1]); y0,y1 = sorted([y0,y1])
                col = (255,0,0,140) if (i%3==0) else ((0,255,0,140) if (i%3==1) else (0,0,255,140))
                dr.rectangle([x0,y0,x1,y1], outline=col[:3], width=3)
            ann = PILImage.alpha_composite(base, over).convert("RGB")
            ann.save(os.path.join(seg_dir, "annotated.png"))
        except Exception as e:
            rospy.logwarn(f"[seg-vis] fail: {e}")

        if raw_text:
            with open(os.path.join(seg_dir, "raw.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text or "")

    def _write_summary(self, out_root: str, note: str, labels: List[str],
                       picks_per_seg: List[int], centroids_rel: List[List[float]],
                       centroids_map: List[List[float]], cloud_frame: str):
        data = {
            "note": note,
            "min_points": self.min_points,
            "labels": labels,
            "picks_per_seg": picks_per_seg,
            "centroids_rel": centroids_rel,
            "centroids_map": centroids_map,
            "cloud_frame": cloud_frame,
            "map_frame": self.map_frame,
            "cam": self.CAM,
            "lidar": self.LIDAR,
        }
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "fusion_summary.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _publish_status(self, msg: str):
        try:
            self.pub_status.publish(String(data=msg))
        except:
            pass

    def _publish_segments(self, segments_out: List[dict]):
        try:
            self.pub_segments.publish(String(data=json.dumps(segments_out)))
        except Exception as e:
            rospy.logwarn(f"[segments] publish failed: {e}")

# --------------- main guard ---------------
if __name__ == "__main__":
    try:
        node = OneShotFusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

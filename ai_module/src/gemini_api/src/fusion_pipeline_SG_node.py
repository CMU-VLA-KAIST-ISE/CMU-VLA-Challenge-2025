#!/opt/conda/envs/python310/bin/python3
# -*- coding: utf-8 -*-

"""
Fusion (One-Shot) Pipeline — prompt-triggered, single Gemini call + dummyVLM waypoint publisher + SceneGraph

- /instruction(std_msgs/String) 수신 시 "딱 1회" 실행
- 또는 /fusion/run_once(Trigger) 호출 시 1회 실행
- LiDAR는 /color_scan_rel, /color_scan_relative 둘 다 구독(둘 중 하나만 있어도 동작)
- 세그 결과(bbox/픽카운트/센트로이드)는 /fusion/segments(String-JSON) 퍼블리시
- 센트로이드(map) PoseArray는 /fusion/centroids_map 퍼블리시 (latched)
- RViz용 MarkerArray는 /fusion/markers 퍼블리시 (latched)
- dummyVLM 주행기 호환: /way_point_with_heading(Pose2D)을 일정 시간 연속 발행 + /resume_navigation 펄스
- SceneGraph(simple + rich): /fusion/scene_graph(String-JSON), /fusion/scene_graph_rich(String-JSON) 퍼블리시 + 덤프

필요 패키지(컨테이너 python310):
  /opt/conda/envs/python310/bin/python3 -m pip install --no-cache-dir \
      numpy==1.26.4 pillow google-genai google-auth google-api-core opencv-python

환경변수:
  export GOOGLE_API_KEY="YOUR_KEY"
"""

import os, io, re, json, time, base64, binascii, threading, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft

from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from std_msgs.msg import String, Empty
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, PoseStamped, Pose2D
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw
from visualization_msgs.msg import Marker, MarkerArray

# Scene graph builder (simple relations)
from scene_graph_builder import build_scene_graph, SceneGraphParams

from google import genai
from google.genai import types


# ---------------------------
# 기본 파라미터
# ---------------------------
DEFAULT_CAMERA = dict(width=1920, height=640, hfov_deg=360.0, vfov_deg=120.0)
DEFAULT_L2C   = dict(x=-0.12, y=-0.075, z=0.265, roll=-1.5707963, pitch=0.0, yaw=-1.5707963)


# ---------------------------
# 유틸
# ---------------------------
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

def cloud2_to_xyz_array(msg: PointCloud2) -> Tuple[np.ndarray, str, rospy.Time]:
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if len(pts) == 0:
        return np.zeros((0,3), dtype=np.float32), (msg.header.frame_id or ""), (msg.header.stamp or rospy.Time(0))
    arr = np.asarray(pts, dtype=np.float32)
    return arr, (msg.header.frame_id or ""), (msg.header.stamp or rospy.Time(0))

def rotmat_rpy(roll, pitch, yaw):
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]], dtype=np.float32)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [ 0,             1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]], dtype=np.float32)
    return Rz @ Ry @ Rx

def scan2pixels_mecanum(laserCloud: np.ndarray, CAM: dict, LIDAR: dict) -> np.ndarray:
    """
    laserCloud: (N,3) lidar/base frame
    CAM: {"x","y","z","roll","pitch","yaw","width","height"}
    LIDAR: {"x","y","z","roll","pitch","yaw"}  # 보통 0
    반환: (N,3) = [u, v, depth_like]
    """
    lidar_offset = np.array([LIDAR.get("x",0.0), LIDAR.get("y",0.0), LIDAR.get("z",0.0)], dtype=np.float32)
    lidarR = rotmat_rpy(LIDAR.get("roll",0.0), LIDAR.get("pitch",0.0), LIDAR.get("yaw",0.0))

    cam_offset = np.array([CAM["x"], CAM["y"], CAM["z"]], dtype=np.float32)
    camR = rotmat_rpy(CAM["roll"], CAM["pitch"], CAM["yaw"])

    xyz = laserCloud[:, :3].astype(np.float32) - lidar_offset
    xyz = xyz @ lidarR
    xyz = xyz - cam_offset
    xyz = xyz @ camR

    W, H = int(CAM["width"]), int(CAM["height"])
    horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2) + 1e-9
    u = (W / (2 * np.pi) * np.arctan2(xyz[:, 0], xyz[:, 2]) + W / 2 + 1).astype(np.int32)
    v = (W / (2 * np.pi) * np.arctan(xyz[:, 1] / (horiDis)) + H / 2 + 1).astype(np.int32)

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
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


# ---------------------------
# Gemini Seg (1회 호출)
# ---------------------------
@dataclass
class SegItem:
    box_2d: List[int]  # [y0,x0,y1,x1] in 0..1000
    mask: str          # data URL or bare base64 PNG
    label: str

class GeminiSegmenter:
    def __init__(self, model="gemini-2.5-flash"):
        self.client = genai.Client()
        self.model_name = model

    def run_once(self, pil_image: PILImage.Image, prompt: Optional[str]=None) -> Tuple[List[SegItem], Optional[str]]:
        if pil_image is None:
            return [], None
        im = pil_image.copy()
        im.thumbnail([1024, 1024], PILImage.Resampling.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="PNG")
        image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

        _prompt = prompt or (
            "You are an expert segmentation model. "
            "Segment large floor obstacles and salient items likely seen by a mobile robot (up to 8). "
            'Return a JSON list with keys: "box_2d" (y0,x0,y1,x1 in 0..1000), "mask" (base64 PNG), "label".'
        )
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=list[SegItem],
        )
        resp = self.client.models.generate_content(model=self.model_name, contents=[_prompt, image_part], config=cfg)

        items = []
        try:
            items = resp.parsed or []
        except Exception:
            pass
        if not items:
            m = re.search(r"```json\s*(.*?)\s*```", getattr(resp, "text", ""), flags=re.S)
            raw = json.loads(m.group(1) if m else getattr(resp, "text", "[]"))
            items = [SegItem(**it) for it in raw]
        return items, getattr(resp, "text", None)


# ---------------------------
# Rich 포맷 헬퍼
# ---------------------------
def _aabb_to_corners(mins, maxs):
    # mins/maxs: [x_min, y_min, z_min] / [x_max, y_max, z_max]
    x0,y0,z0 = mins; x1,y1,z1 = maxs
    # 윗면 4 + 아랫면 4 (샘플 포맷과 순서 맞춤)
    return [
        [x0, y1, z1], [x1, y1, z1], [x1, y0, z1], [x0, y0, z1],
        [x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0],
    ]

_NYU_MAP = {
    # 최소 매핑 예시(필요시 확장)
    "chair":        {"nyu_id":"62","nyu40_id":"5","nyu_label":"chair","nyu40_label":"chair"},
    "sofa":         {"nyu_id":"83","nyu40_id":"6","nyu_label":"sofa","nyu40_label":"sofa"},
    "table":        {"nyu_id":"61","nyu40_id":"9","nyu_label":"table","nyu40_label":"table"},
    "tv":           {"nyu_id":"71","nyu40_id":"15","nyu_label":"tv","nyu40_label":"tv"},
    "potted plant": {"nyu_id":"64","nyu40_id":"12","nyu_label":"plant","nyu40_label":"plant"},
}
def _nyu_lookup(label):
    key = (label or "").lower().strip()
    return _NYU_MAP.get(key, {"nyu_id":"-1","nyu40_id":"-1","nyu_label":key,"nyu40_label":key})


# ---------------------------
# One-Shot Fusion Node
# ---------------------------
class OneShotFusionNode:
    def __init__(self):
        rospy.init_node("fusion_pipeline_node", anonymous=False)
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---- Params
        self.min_points   = rospy.get_param("~min_points", 30)
        self.dump_dir     = rospy.get_param("~dump_dir", "/tmp/fusion_faas")
        self.map_frame    = rospy.get_param("~map_frame", "odom")
        self.prompt_topic = rospy.get_param("~prompt_topic", "/instruction")
        self.seg_prompt   = rospy.get_param("~seg_prompt", "")

        # move_base 방식(옵션)
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.publish_goal_on_single = bool(rospy.get_param("~publish_goal_on_single", False))

        # dummyVLM waypoint 방식(추천)
        self.use_waypoint_interface    = bool(rospy.get_param("~use_waypoint_interface", True))
        self.waypoint_frame            = rospy.get_param("~waypoint_frame", "odom")
        self.waypoint_publish_rate     = float(rospy.get_param("~waypoint_publish_rate", 10.0))
        self.waypoint_publish_duration = float(rospy.get_param("~waypoint_publish_duration", 6.0))
        # NEW: 물체 1개일 때만 웨이포인트 발행
        self.waypoint_only_if_single   = bool(rospy.get_param("~waypoint_only_if_single", True))

        # 간단 프롬프트 쿨다운
        self.cooldown_sec = float(rospy.get_param("~cooldown_sec", 10.0))
        self.last_prompt_time = 0.0

        model_name        = rospy.get_param("~gemini_model", "gemini-2.5-flash")

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
            "x": rospy.get_param("~lidar/x", 0.0),
            "y": rospy.get_param("~lidar/y", 0.0),
            "z": rospy.get_param("~lidar/z", 0.0),
            "roll":  rospy.get_param("~lidar/roll", 0.0),
            "pitch": rospy.get_param("~lidar/pitch", 0.0),
            "yaw":   rospy.get_param("~lidar/yaw", 0.0),
        }

        self.segmenter = GeminiSegmenter(model=model_name)

        # ---- State
        self.lock = threading.Lock()
        self.is_busy = False
        self.last_image: Optional[PILImage.Image] = None
        self.last_cloud_np: Optional[np.ndarray] = None
        self.last_cloud_stamp: Optional[rospy.Time] = None
        self.last_cloud_frame: Optional[str] = None

        # ---- ROS I/O
        self.pub_centroids_map = rospy.Publisher("/fusion/centroids_map", PoseArray, queue_size=1, latch=True)
        self.pub_status        = rospy.Publisher("/fusion/status", String, queue_size=1)
        self.pub_segments      = rospy.Publisher("/fusion/segments", String, queue_size=1)
        self.pub_markers       = rospy.Publisher("/fusion/markers", MarkerArray, queue_size=1, latch=True)
        self.pub_goal          = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)

        # dummyVLM 인터페이스
        self.pub_wp            = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=10)
        self.pub_resume        = rospy.Publisher("/resume_navigation", Empty, queue_size=1, latch=True)
        self.pub_resume_goal   = rospy.Publisher("/resume_navigation_to_goal", Empty, queue_size=1, latch=True)

        # SG 발행 (simple + rich)
        self.pub_scene_graph      = rospy.Publisher("/fusion/scene_graph", String, queue_size=1, latch=True)
        self.pub_scene_graph_rich = rospy.Publisher("/fusion/scene_graph_rich", String, queue_size=1, latch=True)
        self.sg_enable  = bool(rospy.get_param("~sg_enable", True))
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
        self.scene_name  = rospy.get_param("~scene_name", rospy.get_param("/current_scene_name", "unknown_scene"))
        self.region_name = rospy.get_param("~region_name", "default_region")

        # subscribers / service
        self.image_topic = rospy.get_param("~camera_image_topic", "/camera/image")
        self.cloud_topic_a = rospy.get_param("~cloud_topic_a", "/color_scan_rel")
        self.cloud_topic_b = rospy.get_param("~cloud_topic_b", "/color_scan_relative")

        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.cloud_topic_a, PointCloud2, self.cb_cloud, queue_size=1)
        rospy.Subscriber(self.cloud_topic_b, PointCloud2, self.cb_cloud, queue_size=1)
        rospy.Subscriber(self.prompt_topic, String, self.cb_prompt, queue_size=10)

        rospy.Service("/fusion/run_once", Trigger, self.srv_run_once)

        rospy.loginfo("fusion_pipeline_node ready. (one-shot mode)")

    # -------- Callbacks
    def cb_prompt(self, msg: String):
        if time.time() - self.last_prompt_time < self.cooldown_sec:
            rospy.logwarn("[fusion] prompt cooldown; ignored")
            return
        self.last_prompt_time = time.time()

        with self.lock:
            if self.is_busy:
                rospy.logwarn("[fusion] busy; ignoring /instruction")
                return
            if self.last_image is None or self.last_cloud_np is None:
                rospy.logwarn("[fusion] no cached image/cloud; ignoring /instruction")
                return
            self.is_busy = True
        threading.Thread(target=self._run_once_pipeline, args=(msg.data or self.seg_prompt or "",), daemon=True).start()

    def cb_image(self, msg: Image):
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv[..., ::-1]
            self.last_image = PILImage.fromarray(rgb)
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

    # -------- Core (1회 실행)
    def _run_once_pipeline(self, prompt: str):
        ts = now_ts_str()
        out_root = os.path.join(self.dump_dir, ts)
        seg_dir = os.path.join(out_root, "seg");   os.makedirs(seg_dir, exist_ok=True)
        dbg_dir = os.path.join(out_root, "debug"); os.makedirs(dbg_dir, exist_ok=True)

        try:
            # 입력 덤프
            self._dump_inputs(out_root)

            # 1) 세그(정확히 1회)
            try:
                items, raw_text = self.segmenter.run_once(self.last_image, prompt=(prompt or None))
            except Exception as e:
                self._publish_status(f"Seg error: {e}")
                self._write_summary(out_root, "seg_error", [], [], [], [], self.last_cloud_frame or "")
                return

            self._save_seg_artifacts(seg_dir, items, raw_text)
            if not items:
                self._publish_status("Seg returned 0 items")
                self._write_summary(out_root, "seg_empty", [], [], [], [], self.last_cloud_frame or "")
                return

            # 2) 투영
            cloud_rel = self.last_cloud_np
            if cloud_rel is None or cloud_rel.shape[0] == 0:
                self._publish_status("Empty cloud")
                self._write_summary(out_root, "empty_cloud", [], [], [], [], self.last_cloud_frame or "")
                return

            uvd = scan2pixels_mecanum(cloud_rel, self.CAM, self.LIDAR)
            u, v = uvd[:, 0], uvd[:, 1]
            H, W = int(self.CAM["height"]), int(self.CAM["width"])

            # bbox 해석 함수
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

            # 전체 포인트 산포도(디버그)
            try:
                any_pick = np.zeros(u.shape[0], dtype=bool)
                for it in items:
                    bb = _bbox_from_box2d(it.box_2d)
                    if bb is None: continue
                    x0,y0,x1,y1 = bb
                    any_pick |= ((u >= x0) & (u <= x1) & (v >= y0) & (v <= y1))
                draw_points_overlay(self.last_image, u, v, any_pick, os.path.join(dbg_dir, "points_on_image.png"))
            except Exception as e:
                rospy.logwarn(f"[debug] overlay-all failed: {e}")

            centroids_rel, labels, picks_per_seg = [], [], []
            segments_out = []
            obj_pts_rel_list = []  # SG용

            for idx, it in enumerate(items):
                bb = _bbox_from_box2d(it.box_2d)
                if bb is None:
                    picks_per_seg.append(0)
                    segments_out.append({
                        "index": idx, "label": it.label, "box_px": None,
                        "box_px_expanded": None, "picks": 0,
                        "centroid_rel": None, "centroid_map": None,
                    })
                    obj_pts_rel_list.append(None)
                    continue

                x0, y0, x1, y1 = bb
                x0 = max(0, min(W-1, x0)); x1 = max(0, min(W-1, x1))
                y0 = max(0, min(H-1, y0)); y1 = max(0, min(H-1, y1))

                # mask 우선
                cloud_mask = None
                try:
                    mbytes = decode_data_url_to_bytes(it.mask or "")
                    if mbytes:
                        mimg = PILImage.open(io.BytesIO(mbytes)).convert("L")
                        mimg = mimg.resize((max(1, x1-x0), max(1, y1-y0)), PILImage.Resampling.BILINEAR)
                        marr = np.array(mimg)
                        full = np.zeros((H, W), dtype=np.uint8)
                        full[y0:y1, x0:x1] = marr
                        cloud_mask = (full[v, u] > 127)
                except Exception:
                    cloud_mask = None

                # bbox fallback
                if cloud_mask is None:
                    cloud_mask = ((u >= x0) & (u <= x1) & (v >= y0) & (v <= y1))

                idxs = np.where(cloud_mask)[0]

                # 너무 작으면 bbox 10% 팽창 후 재시도
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

                # 디버그: 세그별 픽
                try:
                    pick_mask = np.zeros(u.shape[0], dtype=bool); pick_mask[idxs] = True
                    safe_label = re.sub('[^A-Za-z0-9_-]', '_', it.label)[:40]
                    draw_points_overlay(self.last_image, u, v, pick_mask,
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
                }

                if idxs.size < self.min_points:
                    segments_out.append(seg_info)
                    obj_pts_rel_list.append(None)
                    continue

                pts = cloud_rel[idxs, :]
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
                self._write_summary(out_root, "no_centroids", labels, picks_per_seg, [], [], self.last_cloud_frame or "")
                self._publish_segments(segments_out)
                return

            # 4) TF map 변환
            frame_src = self.last_cloud_frame or "base_link"
            stamp = self.last_cloud_stamp or rospy.Time(0)
            try:
                tfm = self.tf_buffer.lookup_transform(self.map_frame, frame_src, stamp, rospy.Duration(0.5))
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

            # segments_out에도 centroid_map 채워넣기 (순서 매칭)
            map_iter = iter(centroids_map)
            for seg in segments_out:
                if seg["centroid_rel"] is not None:
                    seg["centroid_map"] = next(map_iter, None)

            # 5) Publish PoseArray(map)
            pa = PoseArray()
            pa.header.stamp = rospy.Time.now()
            pa.header.frame_id = self.map_frame
            for cm in centroids_map:
                pose = Pose()
                pose.position = Point(*cm)
                pose.orientation = Quaternion(0,0,0,1)
                pa.poses.append(pose)
            self.pub_centroids_map.publish(pa)

            # === SCENE GRAPH (c) MAP 좌표로 AABB/footprint 계산 + 퍼블리시 ===
            scene_graph = None
            try:
                objects_for_sg = []
                obj_id_running = 0
                for k, seg in enumerate(segments_out):
                    if seg.get("centroid_rel") is None:
                        continue
                    pts_rel = obj_pts_rel_list[k]
                    if pts_rel is None or pts_rel.shape[0] == 0:
                        continue

                    # REL -> MAP
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

                # 로봇 포즈(맵 기준 yaw)
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

                if self.sg_enable and len(objects_for_sg) > 0:
                    scene_graph = build_scene_graph(
                        objects=objects_for_sg,
                        frame=self.map_frame,
                        robot_xyyaw=robot_xyyaw,
                        params=self.sg_params
                    )
                    scene_graph["timestamp"] = float(rospy.Time.now().to_sec())

                    # 파일 저장 (simple)
                    try:
                        with open(os.path.join(out_root, "scene_graph.json"), "w", encoding="utf-8") as f:
                            json.dump(scene_graph, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        rospy.logwarn(f"[scene-graph] save failed: {e}")

                    # 토픽 퍼블리시 (simple)
                    try:
                        self.pub_scene_graph.publish(String(data=json.dumps(scene_graph)))
                    except Exception as e:
                        rospy.logwarn(f"[scene-graph] publish failed: {e}")
            except Exception as e:
                rospy.logwarn(f"[scene-graph] build error: {e}")

            # ---- move_base 목표 자동 발행 (옵션, 단일일 때만)
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

            # ---- WAYPOINT INTERFACE (dummyVLM 호환)
            try:
                # NEW: waypoint_only_if_single 적용
                if self.use_waypoint_interface and (not self.waypoint_only_if_single or len(centroids_map) == 1):
                    gx_map, gy_map, gz_map = centroids_map[0]
                    # map -> waypoint_frame 변환 (필요시)
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

                    # resume 펄스 후 Pose2D 연속 발행
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

            # ---- RViz용 MarkerArray (구체 + 라벨)
            try:
                ma = MarkerArray()
                now = rospy.Time.now()
                ns = "fusion_segments"
                for i, (cm, lbl) in enumerate(zip(centroids_map, labels)):
                    # SPHERE
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

                    # TEXT
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

            # 6) 요약 저장 + 상태 + 세그 전용 토픽
            self._write_summary(out_root, "ok", labels, picks_per_seg, centroids_rel, centroids_map, frame_src)
            status_payload = {"ok": True, "n": len(centroids_map), "labels": labels, "segments": segments_out}
            self._publish_status(json.dumps(status_payload))
            self._publish_segments(segments_out)

            # === RICH 포맷 추가 생성/퍼블리시 ===
            try:
                if scene_graph is not None:
                    # region AABB 전체로 구성
                    all_mins = np.array([+1e9, +1e9, +1e9], dtype=np.float32)
                    all_maxs = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
                    for o in scene_graph.get("objects", []):
                        mn = np.array(o["aabb"]["min"]); mx = np.array(o["aabb"]["max"])
                        all_mins = np.minimum(all_mins, mn); all_maxs = np.maximum(all_maxs, mx)
                    margin = 0.1
                    span = np.maximum(all_maxs - all_mins, margin)
                    all_mins = all_mins - 0.05*span
                    all_maxs = all_maxs + 0.05*span
                    region_bbox = _aabb_to_corners(all_mins.tolist(), all_maxs.tolist())

                    # objects 변환
                    rich_objects = []
                    for o in scene_graph.get("objects", []):
                        mn = np.array(o["aabb"]["min"]); mx = np.array(o["aabb"]["max"])
                        size = (mx - mn)
                        vol  = float(size[0]*size[1]*size[2])
                        nyu  = _nyu_lookup(o["label"])
                        rich_objects.append({
                            "object_id": str(o["id"]),
                            "raw_label": o["label"],
                            "nyu_id": nyu["nyu_id"],
                            "nyu40_id": nyu["nyu40_id"],
                            "nyu_label": nyu["nyu_label"],
                            "nyu40_label": nyu["nyu40_label"],
                            "color_vals": [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                            "color_labels": ["N/A","N/A","N/A"],
                            "color_percentages": ["N/A","N/A","N/A"],
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
                    self.pub_scene_graph_rich.publish(String(data=json.dumps(rich)))
            except Exception as e:
                rospy.logwarn(f"[scene-graph rich] build/publish failed: {e}")

        except Exception as e:
            self._publish_status(f"Exception: {e}")
            self._write_summary(out_root, "exception", [], [], [], [], self.last_cloud_frame or "")
        finally:
            with self.lock:
                self.is_busy = False

    # -------- Dump & Artifacts
    def _dump_inputs(self, out_root: str):
        os.makedirs(out_root, exist_ok=True)
        if self.last_image is not None:
            try:
                self.last_image.save(os.path.join(out_root, "image.png"), "PNG")
            except Exception as e:
                rospy.logwarn(f"[dump] save image failed: {e}")
        if self.last_cloud_np is not None and self.last_cloud_np.size > 0:
            try:
                write_pcd_float32(os.path.join(out_root, "cloud_rel.pcd"),
                                  self.last_cloud_np.astype(np.float32), ["x","y","z"])
            except Exception as e:
                rospy.logwarn(f"[dump] save rel cloud failed: {e}")
            # map 변환 덤프(가능 시)
            frame_src = self.last_cloud_frame or "base_link"
            stamp = self.last_cloud_stamp or rospy.Time(0)
            try:
                tfm = self.tf_buffer.lookup_transform(self.map_frame, frame_src, stamp, rospy.Duration(0.5))
                t = tfm.transform.translation
                q = tfm.transform.rotation
                R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
                tvec = np.array([t.x, t.y, t.z], dtype=np.float32)
                pts_map = self.last_cloud_np.copy()
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
            out.append({"label": it.label, "box_2d": b,
                        "box_px_tryA": [x0a,y0a,x1a,y1a], "box_px_tryB": [x0b,y0b,x1b,y1b],
                        "mask_len": len(it.mask or "")})
        with open(os.path.join(seg_dir, "items.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        try:
            base = self.last_image.convert("RGBA")
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
                f.write(raw_text)

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
        rospy.loginfo("[fusion] %s", msg)
        try:
            self.pub_status.publish(String(data=msg))
        except:
            pass

    def _publish_segments(self, segments_out: List[dict]):
        try:
            self.pub_segments.publish(String(data=json.dumps(segments_out)))
        except Exception as e:
            rospy.logwarn(f"[segments] publish failed: {e}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    try:
        node = OneShotFusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

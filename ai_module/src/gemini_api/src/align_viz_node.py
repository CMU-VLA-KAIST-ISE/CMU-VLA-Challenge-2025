#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align Viz (One-Shot, No Scene Graph)
- /instruction 들어오면 1회 수행 (busy + cooldown 적용)
- /camera/image, /color_scan_rel(/_relative) 구독
- Gemini 세그 → 3D→2D backprojection → 오버레이 이미지 발행/저장
- 상대/절대 PCD 저장으로 정합 진단 용이
- 정지 감지(오도메트리 + TF 폴백)로 자동 1회 실행
"""

import os, io, re, json, time, base64, binascii, threading, math
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft

from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw

from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Odometry  # ✅ 누락 import 추가

# ===== Gemini (google-genai) =====
from google import genai
from google.genai import types


# ---------------- Common utils ----------------
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

def write_pcd_float32(filepath: str, pts: np.ndarray, fields=("x","y","z")):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pts = np.asarray(pts, dtype=np.float32, order="C")
    K = pts.shape[1]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(['4'] * K)}\n"
        f"TYPE {' '.join(['F'] * K)}\n"
        f"COUNT {' '.join(['1'] * K)}\n"
        f"WIDTH {pts.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {pts.shape[0]}\n"
        "DATA binary\n"
    )
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes(order="C"))

def cloud2_to_xyz_array(msg: PointCloud2) -> Tuple[np.ndarray, str, rospy.Time]:
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if len(pts) == 0:
        return np.zeros((0,3), dtype=np.float32), (msg.header.frame_id or ""), (msg.header.stamp or rospy.Time(0))
    arr = np.asarray(pts, dtype=np.float32)
    return arr, (msg.header.frame_id or ""), (msg.header.stamp or rospy.Time(0))

def euler_R(roll, pitch, yaw):
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

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T

def draw_points_overlay(pil_img: PILImage.Image, u: np.ndarray, v: np.ndarray,
                        pick_mask: Optional[np.ndarray], out_path: Optional[str]=None,
                        max_points: int = 80000) -> PILImage.Image:
    W, H = pil_img.size
    idx_all = np.arange(u.shape[0])
    if u.shape[0] > max_points:
        idx_all = np.random.choice(idx_all, size=max_points, replace=False)
    base = pil_img.convert("RGB").copy()
    dr = ImageDraw.Draw(base)
    for i in idx_all[: max_points//2]:
        x, y = int(u[i]), int(v[i])
        if 0 <= x < W and 0 <= y < H:
            dr.point((x, y), fill=(180,180,180))
    if pick_mask is not None and pick_mask.any():
        idx_pick = np.where(pick_mask)[0]
        if idx_pick.size > max_points//2:
            idx_pick = np.random.choice(idx_pick, size=max_points//2, replace=False)
        for i in idx_pick:
            x, y = int(u[i]), int(v[i])
            if 0 <= x < W and 0 <= y < H:
                dr.point((x, y), fill=(255,32,32))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        base.save(out_path, "PNG")
    return base

def _bbox_from_0to1000(box_2d: List[int], W: int, H: int) -> Optional[Tuple[int,int,int,int]]:
    if not (isinstance(box_2d, list) and len(box_2d) == 4):
        return None
    y0a = int(box_2d[0] / 1000.0 * H); x0a = int(box_2d[1] / 1000.0 * W)
    y1a = int(box_2d[2] / 1000.0 * H); x1a = int(box_2d[3] / 1000.0 * W)
    x0b = int(box_2d[0] / 1000.0 * W); y0b = int(box_2d[1] / 1000.0 * H)
    x1b = int(box_2d[2] / 1000.0 * W); y1b = int(box_2d[3] / 1000.0 * H)
    def ok(x0,y0,x1,y1): return (x1>x0) and (y1>y0)
    if ok(x0a,y0a,x1a,y1a): return (x0a,y0a,x1a,y1a)
    if ok(x0b,y0b,x1b,y1b): return (x0b,y0b,x1b,y1b)
    return None

def safe_label(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("", "label", "object", "item"): return "unknown"
    return s

def _get(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

# ---------------- Gemini Segmenter ----------------
# class SegItem(types.Schema):
#     box_2d: List[int]
#     mask: Optional[str]
#     label: str
# (기존) from google.genai import types  # <- 유지해도 됨
# class SegItem(types.Schema): ...         # <- 삭제
# response_schema=List[SegItem]            # <- 삭제

class GeminiSegmenter:
    def __init__(self, model="gemini-2.5-flash"):
        self.client = genai.Client()
        self.model_name = model

    def run_once(self, pil_image, prompt=None, timeout_sec=20.0):
        im = pil_image.copy()
        im.thumbnail([1024, 1024], PILImage.Resampling.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="PNG")
        part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

        _prompt = prompt or (
            "You are an expert segmentation model for mobile robots.\n"
            "Return ONLY a JSON array of objects with keys: box_2d, mask, label.\n"
            "box_2d is [y0,x0,y1,x1] scaled to 0..1000.\n"
            "mask is base64 PNG (may be empty string). label is a short class name.\n"
            "Limit to at most 5 objects."
        )

        # 스키마 없이 호출
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[_prompt, part]
        )
        raw = getattr(resp, "text", "") or "[]"

        # 코드블록 fallback
        m = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.S)
        txt = m.group(1) if m else raw

        try:
            items = json.loads(txt)
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []

        # dict 표준화 (키 보정)
        out = []
        for d in items:
            if not isinstance(d, dict): 
                continue
            out.append({
                "box_2d": d.get("box_2d", []),
                "mask": d.get("mask", "") or "",
                "label": (d.get("label","") or "").strip().lower() or "unknown",
            })
        return out, raw


# ---------------- Projection (single 4x4) ----------------
def project_lidar_to_cam_pixels(
    cloud_xyz: np.ndarray,
    CAM: Dict[str, float],
    LIDAR: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cloud_xyz is None or cloud_xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    # 1) LiDAR->Robot
    R_L = euler_R(LIDAR.get("roll",0.0), LIDAR.get("pitch",0.0), LIDAR.get("yaw",0.0))
    t_L = np.array([LIDAR.get("x",0.0), LIDAR.get("y",0.0), LIDAR.get("z",0.0)], dtype=np.float32)
    T_R_L = make_T(R_L, -R_L @ t_L)  # p_R = R_L*(p_L - t_L)

    # 2) Camera<->Robot
    R_C = euler_R(CAM["roll"], CAM["pitch"], CAM["yaw"])
    t_C = np.array([CAM["x"], CAM["y"], CAM["z"]], dtype=np.float32)
    T_R_C = make_T(R_C, t_C)        # p_R = R_C * p_C + t_C
    T_C_R = np.linalg.inv(T_R_C)    # p_C = R_C^T * (p_R - t_C)

    # 3) LiDAR->Camera
    T_C_L = T_C_R @ T_R_L

    # 4) 적용
    N = cloud_xyz.shape[0]
    homog = np.ones((N,4), dtype=np.float32); homog[:,:3] = cloud_xyz.astype(np.float32)
    cam_pts = (T_C_L @ homog.T).T
    X, Y, Z = cam_pts[:,0], cam_pts[:,1], cam_pts[:,2]

    # 5) Equirectangular
    W = int(CAM["width"]); H = int(CAM["height"])
    vfov_rad = float(CAM["vfov_deg"]) * np.pi / 180.0
    u = (W / (2*np.pi) * np.arctan2(X, Z) + W/2.0 + 1.0)
    hori = np.sqrt(X**2 + Z**2) + 1e-9
    v = (H / vfov_rad * np.arctan(Y / hori) + H/2.0 + 1.0)

    u = np.clip(u, 0, W-1).astype(np.int32)
    v = np.clip(v, 0, H-1).astype(np.int32)
    depth = hori.astype(np.float32)
    return u, v, depth


# ---------------- Main Node ----------------
class AlignVizNode:
    def __init__(self):
        rospy.init_node("align_viz_node", anonymous=False)
        self.bridge = CvBridge()

        # --- Params
        self.dump_root = rospy.get_param("~dump_dir", "/tmp/align_viz")
        os.makedirs(self.dump_root, exist_ok=True)
        self.session_dir = os.path.join(self.dump_root, now_ts_str())
        os.makedirs(self.session_dir, exist_ok=True)

        self.map_frame    = rospy.get_param("~map_frame", "map")
        self.prompt_topic = rospy.get_param("~prompt_topic", "/instruction")
        self.model_name   = rospy.get_param("~gemini_model", "gemini-2.5-flash")
        self.seg_prompt   = rospy.get_param("~seg_prompt", "")

        cam_cfg = rospy.get_param("~camera", {"width":1920, "height":640, "hfov_deg":360.0, "vfov_deg":120.0})
        self.CAM = {
            "width":  int(cam_cfg.get("width", 1920)),
            "height": int(cam_cfg.get("height",640)),
            "hfov_deg": float(cam_cfg.get("hfov_deg", 360.0)),
            "vfov_deg": float(cam_cfg.get("vfov_deg", 120.0)),
            "x": float(rospy.get_param("~l2c/x", -0.12)),
            "y": float(rospy.get_param("~l2c/y", -0.075)),
            "z": float(rospy.get_param("~l2c/z", 0.265)),
            "roll":  float(rospy.get_param("~l2c/roll",  -1.5707963)),
            "pitch": float(rospy.get_param("~l2c/pitch",  0.0)),
            "yaw":   float(rospy.get_param("~l2c/yaw",    -1.5707963)),
        }
        self.LIDAR = {
            "x": float(rospy.get_param("~lidar/x", 0.0)),
            "y": float(rospy.get_param("~lidar/y", 0.0)),
            "z": float(rospy.get_param("~lidar/z", 0.0)),
            "roll":  float(rospy.get_param("~lidar/roll",  0.0)),
            "pitch": float(rospy.get_param("~lidar/pitch", 0.0)),
            "yaw":   float(rospy.get_param("~lidar/yaw",   0.0)),
        }

        self.image_topic  = rospy.get_param("~camera_image_topic", "/camera/image")
        self.cloud_topic_a = rospy.get_param("~cloud_topic_a", "/color_scan_rel")
        self.cloud_topic_b = rospy.get_param("~cloud_topic_b", "/color_scan_relative")

        self.cooldown_sec    = float(rospy.get_param("~cooldown_sec", 10.0))
        self.seg_timeout_sec = float(rospy.get_param("~seg_timeout_sec", 20.0))

        # Pub
        self.pub_overlay_rel = rospy.Publisher("/fusion/overlay_rel", Image, queue_size=1)
        self.pub_overlay_rel_picks = rospy.Publisher("/fusion/overlay_rel_picks", Image, queue_size=1)

        # TF
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(120.0))
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf)

        # --- 정지감지 파라미터
        self.auto_on_stop      = bool(rospy.get_param("~auto_on_stop", True))
        self.odom_topic        = rospy.get_param("~odom_topic", "/odom")
        self.stop_lin_thresh   = float(rospy.get_param("~stop_lin_thresh", 0.02))
        self.stop_ang_thresh   = float(rospy.get_param("~stop_ang_thresh", 0.02))
        self.stop_dwell_sec    = float(rospy.get_param("~stop_dwell_sec", 1.0))
        self.auto_cooldown_sec = float(rospy.get_param("~auto_cooldown_sec", 8.0))

        self.tf_stop_enable    = bool(rospy.get_param("~tf_stop_enable", True))
        self.tf_stop_parent    = rospy.get_param("~tf_stop_parent", "map")
        self.tf_stop_child     = rospy.get_param("~tf_stop_child",  "base_link")

        # --- 상태
        self._stop_since: Optional[float] = None
        self._auto_last_fire: float = 0.0
        self._tf_stop_last = None
        self._lock = threading.Lock()
        self.is_busy = False
        self.last_fire = 0.0

        if self.auto_on_stop:
            rospy.Subscriber(self.odom_topic, Odometry, self._cb_odom, queue_size=20)
            if self.tf_stop_enable:
                self._tf_timer = rospy.Timer(rospy.Duration(0.1), self._cb_tf_stop_poll)

        # State (buffers)
        self.last_image: Optional[PILImage.Image] = None
        self.last_image_np: Optional[np.ndarray] = None
        self.last_cloud_np: Optional[np.ndarray] = None
        self.last_cloud_stamp: Optional[rospy.Time] = None
        self.last_cloud_frame: Optional[str] = None

        self.segmenter = GeminiSegmenter(model=self.model_name)

        # Sub
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.cloud_topic_a, PointCloud2, self.cb_cloud, queue_size=1)
        rospy.Subscriber(self.cloud_topic_b, PointCloud2, self.cb_cloud, queue_size=1)
        rospy.Subscriber(self.prompt_topic, String, self.cb_prompt, queue_size=10)

        rospy.loginfo("align_viz_node ready.")

    # ------------- Auto-stop
    def _have_fresh_inputs(self) -> bool:
        return (self.last_image is not None) and (self.last_cloud_np is not None)

    def _auto_stop_logic(self, lin: float, ang: float):
        now = time.time()
        stopped = (lin <= self.stop_lin_thresh) and (ang <= self.stop_ang_thresh)
        if stopped:
            if self._stop_since is None:
                self._stop_since = now
            dwell = now - self._stop_since
            if dwell >= self.stop_dwell_sec:
                if (now - self._auto_last_fire) >= self.auto_cooldown_sec:
                    with self._lock:
                        if not self.is_busy and self._have_fresh_inputs():
                            self.is_busy = True
                            self._auto_last_fire = now
                            rospy.loginfo(f"[align_viz] auto: stopped for {dwell:.2f}s → run segmentation once")
                            threading.Thread(target=self._run_seg_once, daemon=True).start()
        else:
            self._stop_since = None

    def _cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)
        lin = math.hypot(vx, vy)
        ang = abs(wz)
        self._auto_stop_logic(lin, ang)

    def _cb_tf_stop_poll(self, _evt):
        try:
            tfm = self.tf_buf.lookup_transform(self.tf_stop_parent, self.tf_stop_child, rospy.Time(0), rospy.Duration(0.2))
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

    def _run_seg_once(self):
        try:
            # 자동 트리거 경로는 내부 prompt 사용
            self.run_once(self.seg_prompt or "")
        except Exception as e:
            rospy.logwarn(f"[align_viz] seg once failed: {e}")
        finally:
            with self._lock:
                self.is_busy = False

    # ------------- Manual trigger
    def cb_image(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv_bgr[..., ::-1].copy()
            self.last_image_np = rgb
            self.last_image = PILImage.fromarray(rgb)
        except Exception as e:
            rospy.logwarn(f"[image] cv_bridge failed: {e}")

    def cb_cloud(self, msg: PointCloud2):
        arr, frame_id, stamp = cloud2_to_xyz_array(msg)
        self.last_cloud_np = arr
        self.last_cloud_stamp = stamp
        self.last_cloud_frame = frame_id or "base_link"

    def cb_prompt(self, msg: String):
        if self.is_busy:
            rospy.logwarn("[align] busy; ignore /instruction")
            return
        if time.time() - self.last_fire < self.cooldown_sec:
            rospy.logwarn("[align] cooldown; ignore /instruction")
            return
        if self.last_image is None or self.last_cloud_np is None:
            rospy.logwarn("[align] no cache image/cloud")
            return
        self.is_busy = True
        self.last_fire = time.time()
        threading.Thread(
            target=self.run_once,
            args=(msg.data or self.seg_prompt or "",),
            daemon=True
        ).start()

    # ------------- Core
    def run_once(self, prompt: str):
        session = self.session_dir
        os.makedirs(session, exist_ok=True)
        try:
            # 입력 덤프
            if self.last_image is not None:
                self.last_image.save(os.path.join(session, "image.png"), "PNG")
            if self.last_cloud_np is not None and self.last_cloud_np.size > 0:
                write_pcd_float32(os.path.join(session, "cloud_rel.pcd"), self.last_cloud_np, ("x","y","z"))
                self.dump_map_cloud(session)

            # 1) 세그
            try:
                items, raw = self.segmenter.run_once(self.last_image, prompt=prompt, timeout_sec=self.seg_timeout_sec)
            except Exception as e:
                rospy.loginfo(f"[align] seg error: {e}")
                self.is_busy = False
                return

            with open(os.path.join(session, "seg_raw.txt"), "w", encoding="utf-8") as f:
                f.write(raw or "")

            # 2) Back-projection
            u, v, depth = project_lidar_to_cam_pixels(self.last_cloud_np, self.CAM, self.LIDAR)
            W = int(self.CAM["width"]); H = int(self.CAM["height"])

            # 전체 오버레이
            base_overlay = draw_points_overlay(
                self.last_image, u, v, pick_mask=None,
                out_path=os.path.join(session, "overlay_rel_all.png")
            )
            self.pub_overlay_rel.publish(self.bridge.cv2_to_imgmsg(np.array(base_overlay)[..., ::-1], encoding="bgr8"))

            # 세그 기반 픽마스크
            pick_any = np.zeros(u.shape[0], dtype=bool)
            ann = self.last_image.convert("RGBA")
            over = PILImage.new("RGBA", ann.size, (0,0,0,0))
            dr = ImageDraw.Draw(over)

            for i, it in enumerate(items):
                #lbl = safe_label(getattr(it, "label", "unknown"))
                lbl = (_get(it, "label", "unknown") or "unknown").strip().lower()
                b2d = _get(it, "box_2d", [])
                mbytes = decode_data_url_to_bytes(_get(it, "mask", "") or "")
                bb = _bbox_from_0to1000(b2d, W, H)
                if bb is None:
                    continue
                x0,y0,x1,y1 = bb

                mask_idx = None
                try:
                    mbytes = decode_data_url_to_bytes(getattr(it, "mask","") or "")
                    if mbytes:
                        mimg = PILImage.open(io.BytesIO(mbytes)).convert("L")
                        mimg = mimg.resize((max(1, x1-x0), max(1, y1-y0)), PILImage.Resampling.BILINEAR)
                        marr = np.array(mimg)
                        full = np.zeros((H, W), dtype=np.uint8)
                        full[y0:y1, x0:x1] = marr
                        mask_idx = (full[v, u] > 127)
                except Exception:
                    mask_idx = None

                if mask_idx is None:
                    mask_idx = ((u >= x0) & (u <= x1) & (v >= y0) & (v <= y1))

                pick_any |= mask_idx

                col = (255,0,0,160) if (i%3==0) else ((0,255,0,160) if (i%3==1) else (0,0,255,160))
                dr.rectangle([x0,y0,x1,y1], outline=col[:3], width=3)

                tx, ty = x0, max(0, y0-18)
                dr.rectangle([tx, ty, tx+8*len(lbl)+10, ty+18], fill=(0,0,0,160))
                d2 = ImageDraw.Draw(over)
                d2.text((tx+5, ty+2), lbl, fill=(255,255,255,255))

            ann = PILImage.alpha_composite(ann, over).convert("RGB")
            picks_overlay = draw_points_overlay(
                ann, u, v, pick_mask=pick_any,
                out_path=os.path.join(session, "overlay_rel_picks.png")
            )
            self.pub_overlay_rel_picks.publish(self.bridge.cv2_to_imgmsg(np.array(picks_overlay)[..., ::-1], encoding="bgr8"))

            summary = {
                "note": "align_viz_ok",
                "n_points": int(self.last_cloud_np.shape[0]),
                "cam": self.CAM,
                "lidar": self.LIDAR,
                "map_frame": self.map_frame,
                "cloud_frame": self.last_cloud_frame or "",
                "labels": [safe_label(getattr(it, "label", "unknown")) for it in items],
            }
            with open(os.path.join(session, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            rospy.loginfo("[align] done. overlays & PCDs dumped at %s", session)
        finally:
            self.is_busy = False

    def dump_map_cloud(self, session_dir: str):
        if self.last_cloud_np is None or self.last_cloud_np.size == 0:
            return
        frame_src = self.last_cloud_frame or "base_link"
        stamp = self.last_cloud_stamp or rospy.Time(0)
        try:
            try:
                tfm = self.tf_buf.lookup_transform(self.map_frame, frame_src, stamp, rospy.Duration(0.5))
            except Exception:
                tfm = self.tf_buf.lookup_transform(self.map_frame, frame_src, rospy.Time(0), rospy.Duration(0.5))
            t = tfm.transform.translation
            q = tfm.transform.rotation
            R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            tvec = np.array([t.x, t.y, t.z], dtype=np.float32)
            pts = (R @ self.last_cloud_np.T).T + tvec
            write_pcd_float32(os.path.join(session_dir, "cloud_map.pcd"), pts, ("x","y","z"))
        except Exception as e:
            rospy.logwarn(f"[align] map transform failed: {e}")


def main():
    AlignVizNode()
    rospy.spin()

if __name__ == "__main__":
    main()

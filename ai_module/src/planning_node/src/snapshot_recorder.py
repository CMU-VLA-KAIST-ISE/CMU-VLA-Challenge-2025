#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from scipy.spatial.transform import Rotation as rot
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_msgs.msg import Header, Bool
import numpy as np
import cv2
import math


def ts_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

class SnapshotRecorder:
    def __init__(self):
        rospy.init_node("snapshot_recorder", anonymous=True)

        # ---- params ----
        gp = rospy.get_param
        self.interval_sec = float(rospy.get_param("~interval_sec", 2.0))
        self.require_all_streams = bool(rospy.get_param("~require_all_streams", False))
        self.require_pose = bool(rospy.get_param("~require_pose", True))
        self.save_dir = os.path.expanduser(rospy.get_param("~save_dir", "~/CMU-VLA-Challenge/snapshots"))
        self.scene_name = rospy.get_param("/current_scene_name", "unknown_scene")

        self.color_topic = rospy.get_param("~color_topic", "/camera/image/compressed")
        self.color_is_compressed = bool(rospy.get_param("~color_is_compressed", True))
        self.lidar_topic = rospy.get_param("~lidar_topic", "/color_scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/state_estimation")
        self.done_topic = rospy.get_param("~done_topic", "/exploration_done")

        # LiDAR 옵션
        self.lidar_is_world = bool(gp("~lidar_is_world", True))
        self.lidar_downsample = float(rospy.get_param("~lidar_downsample", 1.0))
        self.lidar_max_points = int(rospy.get_param("~lidar_max_points", 120000))
        self.dedup_exact = bool(rospy.get_param("~dedup_exact", True))
        self.merge_chunk_limit = int(rospy.get_param("~merge_chunk_limit", 800000))

        # Depth 카메라 모델
        self.W = int(rospy.get_param("~width", 1920))
        self.H = int(rospy.get_param("~height", 640))
        self.depth_min_m = float(rospy.get_param("~depth_min_m", 1.0))
        self.depth_max_m = float(rospy.get_param("~depth_max_m", 500.0))
        self.camera_offset_z = float(gp("~camera_offset_z", 0.0))

        # 출력 퍼블리시(옵션)
        self.publish_depth = bool(rospy.get_param("~publish_depth", False))
        self.depth_topic = rospy.get_param("~depth_topic", "/fused/depth_image")
        self.pub_depth = rospy.Publisher(self.depth_topic, Image, queue_size=1) if self.publish_depth else None
        self.bridge = CvBridge()

        # ---- dirs ----
        session = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.save_dir, f"scene={self.scene_name}", session)
        self.dir_color = os.path.join(base, "color")
        self.dir_depth = os.path.join(base, "depth")
        self.dir_lidar = os.path.join(base, "lidar")
        ##디버깅용 상대좌표계로 표현된 lidar 저장 위치
        self.dir_lidar_base = os.path.join(base, "lidar_base")
        for d in (self.dir_color, self.dir_depth, self.dir_lidar, self.dir_lidar_base):
            os.makedirs(d, exist_ok=True)
        rospy.loginfo(f"[snapshot] saving to: {base}")

        # ---- state ----
        self.seq = 0
        self.last_color_msg = None
        self.last_odom_msg = None
        self._already_shutdown = False
        self._buf_world = []
        self.merge_points_total = [] # list of (N,6)
        self._merged_total_count = 0

        # ---- subscribers ----
        if self.color_is_compressed:
            rospy.Subscriber(self.color_topic, CompressedImage, self.cb_color, queue_size=10)
        else:
            rospy.Subscriber(self.color_topic, Image, self.cb_color_raw, queue_size=10)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.cb_lidar, queue_size=5)
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=50)
        rospy.Subscriber(self.done_topic, Bool, self.cb_done, queue_size=1)

        # ---- timer ----
        if self.interval_sec > 0.0:
            rospy.Timer(rospy.Duration(self.interval_sec), self.on_timer)
        else:
            self.timer=None

        # intrinsics 준비
        rospy.on_shutdown(self._on_shutdown_merge_save)
    
    #--depth 이미지 생성을 위한 helper function 모음--#

    ##위에서 받은 로봇의 위상에서 회전에 대한 부분이 쿼터니언으로 제공됨 ->이를 3차원 좌표계(회전 행렬)로 전환
    @staticmethod
    def _quat_to_R(x, y, z, w):
        q = [x,y,z,w]
        r=rot.from_quat(q)
        R=r.as_dcm()
        return R

    ##위에서 만든 회전행렬을 바탕으로 (거리,시야각)으로 된 lidar 정보를 rmsid 3차원 x,y,z 좌표계로 전환하는 함수
    @staticmethod
    def _apply_RT(R, t, pts):
        """(N,3) pts(base) → world: R_wb @ pts + t_wb"""
        return (pts@R.T)+t[None, :]

    ##위의 함수의 역함수
    @staticmethod
    def _apply_RT_inv(R, t, pts_world):
        """(N,3) pts(world) → base(now): R_bw @ (pts - t_wb)"""
        R_bw = R.T
        return (pts_world-t[None, :])@R_bw.T

    ##일단 가장 최근의 pose 상태를 알아야함
    def _current_RT_wb(self):
        if self.last_odom_msg is None:
            return None
        p=self.last_odom_msg.pose.pose.position
        q=self.last_odom_msg.pose.pose.orientation
        R=self._quat_to_R(q.x,q.y,q.z,q.w)
        t=np.array([p.x, p.y, p.z], dtype=np.float32)
        return R,t
    # ---------- callbacks ----------
    def cb_odom(self, msg: Odometry):
        self.last_odom_msg = msg
    def cb_color(self, msg: CompressedImage):
        self.last_color_msg = msg
    def cb_color_raw(self, msg: Image):
        self.last_color_msg = msg

    def cb_lidar(self, msg: PointCloud2):
        if self._already_shutdown:
            return
        try:
            ##여기서 pose 최신화
            rt = self._current_RT_wb()
            if rt is None and self.require_pose:
                rospy.logwarn_throttle(5.0, "[snapshot] pose not ready; skipping lidar frame")
                return
            ##여기서 R_wb는 orientatino 정보(R로 표현), t_wb는 position 정보
            R_wb, t_wb = rt

            fields = [f.name for f in msg.fields]
            color_field = 'rgb' if 'rgb' in fields else ('rgba' if 'rgba' in fields else None)
            names = ("x","y","z",color_field) if color_field else ("x","y","z")

            rows = []
            for p in pc2.read_points(msg, field_names=names, skip_nans=True):
                if color_field:
                    x,y,z,c = p
                    c_uint = np.array([np.float32(c)], dtype=np.float32).view(np.uint32)[0]
                    r = (c_uint >> 16) & 0xFF; g = (c_uint >> 8) & 0xFF; b = c_uint & 0xFF
                    rows.append((x,y,z,r,g,b))
                else:
                    x,y,z = p
                    rows.append((x,y,z,128,128,128))
            if not rows:
                return
            pts = np.asarray(rows, dtype=np.float32)  # (N,6)

            # 다운샘플/리밋
            ds = self.lidar_downsample
            if 0 < ds < 1.0:
                step = max(1, int(round(1.0 / ds)))
                pts = pts[::step]
            cap = self.lidar_max_points
            if cap and len(pts) > cap:
                idx = np.random.choice(len(pts), cap, replace=False)
                pts = pts[idx]

            ##이제 좌표계 전환을 해야함 : 로봇의 위치를 고려한 world 좌표로 전환 -> 애초에 Lidar가 절대 좌표계로 전달 
            #xyz_b = pts[:, :3]
            #rgb = pts[:, 3:6]
            #xyz_w=self._apply_RT(R_wb, t_wb, xyz_b)
            #pts_world=np.concatenate([xyz_w, rgb], axis=1)

            ##중복된 좌표는 제거(만약에 조금 noise 제거를 원한다면, 여기 부분을 clustering으로 변환해야함)
            #if self.dedup_exact:
                #pts_world = self._exact_dedup_xyz(pts_world)

            # 2초 버퍼 적재
            self._buf_world.append(pts)
            self._append_total_merge(pts)
        except Exception as e:
            rospy.logerr(f"[snapshot] lidar buffer error: {e}")

    def cb_done(self, _msg: Bool):
        rospy.loginfo("[snapshot] /exploration_done received → stop & merge-save")
        self._merge_and_exit()
    # ---------- timer (atomic snapshot) ----------
    def on_timer(self, _evt):
        if self._already_shutdown:
            return
        # 준비 체크
        pose_ready = (self._current_RT_wb() is not None) or (not self.require_pose)
        if self.require_all_streams and (self.last_color_msg is None or len(self._buf_world) == 0):
            rospy.logwarn_throttle(5.0, "[snapshot] waiting for color+lidar before first snapshot...")
            return
        if len(self._buf_world) == 0:
            return
        if not pose_ready:
            rospy.logwarn_throttle(5.0, "[snapshot] pose not ready; skipping snapshot tick")
            return

        # 공통 base 이름 (동시 저장 보장)
        base = f"snap_{self.seq:06d}_{ts_str()}"

        try:
            # 1) 2초마다 모은 world 좌표들 합치기 + 여기서도 재중복 제거
            pts6_world = np.concatenate(self._buf_world, axis=0).astype(np.float32, copy=False)
            self._buf_world.clear()
            #if self.dedup_exact:
                #pts6_world = self._exact_dedup_xyz(pts6_world)
            
            # 2) 여기서 모은 좌표들은 world 기준임 : 현재 로봇이 보는 라이더 점들은 다르기 때문에 역과정으로 다시 로봇이 보는 lidar 정보로 전환
            R_wb_now, t_wb_now=self._current_RT_wb() if self._current_RT_wb() is not None else (np.eye(3, dtype=np.float32),np.zeros(3, dtype=np.float32))
            
            xyz_now_base = self._apply_RT_inv(R_wb_now, t_wb_now, pts6_world[:, :3])
            rgb_world = pts6_world[:, 3:6]
            pts6_base = np.concatenate([xyz_now_base, rgb_world], axis=1)

            depth = self._project_to_depth_spherical_like_cpp(pts6_base[:, :3])

            # 3) 저장
            # 3-1) Color
            if self.last_color_msg is not None:
                color_path = os.path.join(self.dir_color, base + ".png")
                self._save_png_from_msg(self.last_color_msg, color_path)
            else:
                rospy.logwarn_throttle(5.0, "[snapshot] no color yet; skipping color save this tick")

            # 3-2) Depth (32FC1 m -> 16UC1 mm)
            depth_path = os.path.join(self.dir_depth, base + ".png")
            self._save_depth_array_mm_png(depth, depth_path)

            # 3-3) LiDAR PLY (xyzrgb)
            lidar_world_path = os.path.join(self.dir_lidar, base + ".ply")
            self._save_ply_xyzrgb(pts6_world, lidar_world_path)

            lidar_base_path = os.path.join(self.dir_lidar_base, base + ".ply")
            self._save_ply_xyzrgb(pts6_base, lidar_base_path)

            #rviz 확인용 depth publish
            if self.publish_depth and self.pub_depth is not None:
                self._publish_depth(depth)

            self.seq += 1
            rospy.loginfo(f"[snapshot] saved: {base}")
        except Exception as e:
            rospy.logerr(f"[snapshot] snapshot failed: {e}")

    # ---------- helpers: IO ----------
    def _save_png_from_msg(self, msg, out_path: str):
        if isinstance(msg, CompressedImage):
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(out_path, img)

    @staticmethod
    def _save_ply_xyzrgb(pts6: np.ndarray, out_path: str):
        if pts6.size == 0:
            return
        with open(out_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {pts6.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for x, y, z, r, g, b in pts6:
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")

    def _save_depth_array_mm_png(self,depth_m: np.ndarray, out_path: str):
        d = np.asarray(depth_m, dtype=np.float32)
        mm = np.zeros_like(d, dtype=np.uint16)
        valid = np.isfinite(d) & (d > 0.0)
        ##실제 점을 gray_scale로 거리에 따라 표현 가까울수록 밝고 멀수록 어둡게. 
        dmin = max(1e-6, float(self.depth_min_m))
        dmax = max(dmin + 1e-6, float(self.depth_max_m))
        ##이 거리에 따라 0과 1로 정규화
        norm = np.zeros_like(d, dtype=np.float32)
        if np.any(valid):
            nv=(d[valid]-dmin)/(dmax-dmin)
            nv=np.clip(nv,0.0,1.0)
            norm[valid] = 1.0 - nv
        mm=(norm*255.0+0.5).astype(np.uint8)
        ##빈 공간이 너무 많은데 빈 공간을 검은색으로 찍으면 실제 이미지가 거의 안 보여서 빈 공간을 흰색으로 표현
        mm[~valid] =255
        cv2.imwrite(out_path, mm)

    def _publish_depth(self, depth: np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera"
        self.pub_depth.publish(msg)

    # ---------- helpers: LiDAR/Depth math ----------
    ##color scan generation에서 따온 코드
    def _project_to_depth_spherical_like_cpp(self, pts_base: np.ndarray) -> np.ndarray:
        """C++ 예제의 방식대로 파노라믹 depth 생성.
        입력: pts_base (N,3), 현재 시점의 로봇 base 프레임(X forward, Y left, Z up 가정)
        투영:
        u = -W/(2π) * atan2(y, x) + W/2
        v = -W/(2π) * atan(z / sqrt(x^2 + y^2)) + H/2
        깊이: euclidean range = sqrt(x^2 + y^2 + z^2) (C++는 색상 샘플링이라 깊이정의 없음 → 여기선 range 채택)
        가장 가까운 포인트를 픽셀에 기록.
        """
        depth = np.zeros((self.H, self.W), dtype=np.float32)
        if pts_base.size == 0:
            return depth   
        X = pts_base[:, 0].astype(np.float32)
        Y = pts_base[:, 1].astype(np.float32)
        Z = pts_base[:, 2].astype(np.float32) - np.float32(self.camera_offset_z)
        # 범위/마스크
        rng = np.sqrt(X * X + Y * Y + Z * Z)
        hori = np.sqrt(X * X + Y * Y)
        # 깊이 범위 필터
        mask = (rng > self.depth_min_m) & (rng < self.depth_max_m)
        if not np.any(mask):
            return depth
        X = X[mask]; Y = Y[mask]; Z = Z[mask]; rng = rng[mask]; hori = hori[mask]
        # 파노라믹 인덱스(C++ 로직과 동일하게 width 기준 스케일 사용)
        # NOTE: +1 보정은 1-based 인덱스에서 비롯 — 여기서는 0-based이므로 생략하고 클리핑
        u = (-self.W / (2.0 * math.pi) * np.arctan2(Y, X) + self.W / 2.0).astype(np.int32)
        v = (-self.W / (2.0 * math.pi) * np.arctan2(Z, np.maximum(hori, 1e-6)) + self.H / 2.0).astype(np.int32)
        inb = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        if not np.any(inb):
            return depth
        u = u[inb]; v = v[inb]; rng = rng[inb]
        # 픽셀마다 가장 가까운 range를 선택(먼저 pixel-id→range로 정렬 후 unique)
        linear = v * self.W + u
        order = np.lexsort((rng, linear)) # pixel → range(오름차순)
        linear = linear[order]; rng = rng[order]
        uniq, first = np.unique(linear, return_index=True)
        vv = (uniq // self.W).astype(np.int32)
        uu = (uniq % self.W).astype(np.int32)
        depth[vv, uu] = rng[first]
        return depth
    # ---------- session-wide merge & graceful stop ----------
    def _append_total_merge(self, pts6: np.ndarray):
        if pts6.size == 0:
            return
        self.merge_points_total.append(pts6)
        self._merged_total_count += pts6.shape[0]
        if self._merged_total_count > self.merge_chunk_limit and len(self.merge_points_total) > 1:
            merged_now = np.vstack(self.merge_points_total)
            #merged_now = self._exact_dedup_xyz(merged_now)
            self.merge_points_total = [merged_now]
            self._merged_total_count = merged_now.shape[0]


    def _save_merged_lidar(self):
        if not self.merge_points_total:
            rospy.logwarn("[snapshot] no session-wide lidar to merge")
            return
        merged = np.vstack(self.merge_points_total) if len(self.merge_points_total) > 1 else self.merge_points_total[0]
        #merged = self._exact_dedup_xyz(merged)
        # 저장
        np.save(os.path.join(self.dir_lidar, "merged_lidar_color.npy"), merged)
        ply_path = os.path.join(self.dir_lidar, "merged_lidar_color.ply")
        self._save_ply_xyzrgb(merged, ply_path)
        rospy.loginfo(f"[snapshot] merged lidar saved: {merged.shape[0]} pts")


    def _merge_and_exit(self):
        if self._already_shutdown:
            return
        self._already_shutdown = True
        try:
            if self.timer is not None:
                self.timer.shutdown()
        except Exception:
            pass
        try:
            self._save_merged_lidar()
        finally:
            rospy.signal_shutdown("exploration_done")


    def _on_shutdown_merge_save(self):
        if not self._already_shutdown:
            try:
                self._save_merged_lidar()
            except Exception as e:
                rospy.logwarn(f"[snapshot] merge-on-shutdown failed: {e}")

if __name__ == "__main__":
    try:
        SnapshotRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import rospy
from nav_msgs.msg import Odometry
import tf.transformations as tft

# ===== 고정 설정 =====
ODOM_TOPIC       = "/state_estimation"
SAMPLE_PERIOD_S  = 1.0
USE_HEADER_STAMP = True   # msg.header.stamp 우선
APPEND_MODE      = False  # 보통 False, true면 이어쓰기
# =====================

def _find_project_root(start_dir: str) -> str:
    """
    위로 올라가며 'data' 폴더가 있는 최상위(프로젝트 루트)를 찾음.
    못 찾으면 start_dir 반환.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)

def _next_numbered_path(data_dir: str) -> str:
    """
    data 디렉터리 안에서 '1.txt', '2.txt', ... 형식의 다음 파일명을 결정.
    """
    os.makedirs(data_dir, exist_ok=True)
    existing = [f for f in os.listdir(data_dir) if re.match(r"^\d+\.txt$", f)]
    if not existing:
        idx = 1
    else:
        nums = [int(re.match(r"^(\d+)\.txt$", f).group(1)) for f in existing]
        idx = max(nums) + 1
    return os.path.join(data_dir, f"{idx}.txt")

class ActualTrajRecorder:
    """
    x y z roll pitch yaw time 형식으로 SAMPLE_PERIOD_S마다 기록.
    파일은 프로젝트 루트의 data/#.txt로 자동 생성.
    """
    def __init__(self):
        rospy.init_node("actual_traj_recorder", anonymous=False)

        # 프로젝트 루트 & 출력 경로
        script_dir = os.path.dirname(os.path.abspath(__file__))
        proj_root  = _find_project_root(script_dir)
        data_dir   = os.path.join(proj_root, "data")
        self.output_path = _next_numbered_path(data_dir)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        mode = "a" if (APPEND_MODE and os.path.exists(self.output_path)) else "w"
        self.fout = open(self.output_path, mode, buffering=1)

        self.latest = None  # (x,y,z, roll,pitch,yaw, t)

        rospy.Subscriber(ODOM_TOPIC, Odometry, self._cb_odom, queue_size=50)
        self.timer = rospy.Timer(rospy.Duration(SAMPLE_PERIOD_S), self._on_timer, oneshot=False)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("[actual_traj_recorder] output: %s", self.output_path)
        rospy.loginfo("[actual_traj_recorder] topic:  %s", ODOM_TOPIC)
        rospy.loginfo("[actual_traj_recorder] period: %.2fs", SAMPLE_PERIOD_S)

    def _cb_odom(self, msg: Odometry):
        # 시간
        if USE_HEADER_STAMP and msg.header.stamp.to_sec() > 0:
            t = msg.header.stamp.to_sec()
        else:
            t = rospy.Time.now().to_sec()

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        try:
            roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        except Exception:
            roll = pitch = yaw = 0.0

        self.latest = (float(p.x), float(p.y), float(p.z),
                       float(roll), float(pitch), float(yaw),
                       float(t))

    def _on_timer(self, _event):
        if self.latest is None:
            return
        x, y, z, r, p, yw, t = self.latest
        self.fout.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {p:.6f} {yw:.6f} {t:.6f}\n")

    def _on_shutdown(self):
        try:
            if self.fout:
                self.fout.flush()
                self.fout.close()
                rospy.loginfo("[actual_traj_recorder] file closed: %s", self.output_path)
        except Exception:
            pass

if __name__ == "__main__":
    try:
        ActualTrajRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
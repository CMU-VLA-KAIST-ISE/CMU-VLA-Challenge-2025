#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import json
import random
import numpy as np
import cv2
from datetime import datetime
from collections import deque
import tf.transformations as tft
import time
from nav_msgs.msg import OccupancyGrid
import math
from heapq import heappush, heappop
import traceback

# ROS 메시지 타입 임포트
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Empty
import sensor_msgs.point_cloud2 as pc2

# 변환 및 저장을 위한 라이브러리
from cv_bridge import CvBridge

# ----- 상수 정의 -----
ACTION_NORTH = 0
ACTION_EAST = 1
ACTION_SOUTH = 2
ACTION_WEST = 3
ACTION_MAP = {
    0: "NORTH",
    1: "EAST",
    2: "SOUTH",
    3: "WEST"
}

class MissionNode:
    def __init__(self):
        rospy.init_node('graph_mission_node', anonymous=True)

        # ----- 설정 변수 -----
        self.waypoint_reach_dis = 0.28
        self.min_wp_spacing = 1.2
        self.min_start_wp_dist = 1.0
        self.carrot_min = 1.0
        self.carrot_max = 1.4

        # ----- 상태 변수 -----
        self.vehicle_x, self.vehicle_y, self.vehicle_yaw = 0.0, 0.0, 0.0
        self.missions_data = None
        self.graph_data = None
        self.current_scene = "unknown_scene" # [추가]

        # 데이터 수집 관련
        self.save_base_path = "/home/kwakjs/CMU-VLA-Challenge/collected_data/"
        self.latest_image_msg, self.latest_lidar_msg = None, None
        self.bridge = CvBridge()

        # ----- ROS Subscriber & Publisher -----
        rospy.Subscriber("/state_estimation", Odometry, self.pose_handler, queue_size=10)
        rospy.Subscriber("/camera/image", Image, self.image_callback, queue_size=10)
        rospy.Subscriber("/color_scan", PointCloud2, self.lidar_callback, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/way_point_with_heading", Pose2D, queue_size=5)
        self.resume_pub = rospy.Publisher("/resume_navigation", Empty, queue_size=1, latch=True)
        self.resume_goal_pub = rospy.Publisher("/resume_navigation_to_goal", Empty, queue_size=1, latch=True)

    def pose_handler(self, msg):
        """로봇 위치 업데이트"""
        self.vehicle_x, self.vehicle_y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        (_, _, self.vehicle_yaw) = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

    def image_callback(self, msg):
        self.latest_image_msg = msg

    def lidar_callback(self, msg):
        self.latest_lidar_msg = msg

    def load_missions_and_graph(self):
        """[수정] 현재 씬에 맞는 _waypoints.json과 그래프 데이터 로드"""
        base_path = "/home/kwakjs/CMU-VLA-Challenge"
        
        # [수정] 현재 씬 이름을 self.current_scene에 저장
        self.current_scene = rospy.get_param("~scene_name", "") or rospy.get_param("/current_scene_name", "unknown_scene_fallback")
        
        # [수정] missions.json 대신 {scene}_waypoints.json 파일 경로 생성
        missions_file = os.path.join(base_path, f"system/unity/src/vehicle_simulator/mesh/unity/{self.current_scene}/{self.current_scene}_waypoints.json")
        graph_file = "/tmp/graph_data.json"

        try:
            # missions.json 로드
            if os.path.exists(missions_file):
                with open(missions_file, 'r') as f:
                    self.missions_data = json.load(f)
                rospy.loginfo(f"Loaded {len(self.missions_data.get('missions', []))} missions for scene '{self.current_scene}' from {missions_file}")
            else:
                rospy.logerr(f"Missions file not found: {missions_file}")
                return False

            # 그래프 데이터 로드 (GraphBuilder가 생성한 파일)
            if os.path.exists(graph_file):
                with open(graph_file, 'r') as f:
                    self.graph_data = json.load(f)
                rospy.loginfo("Graph data loaded successfully")
            else:
                rospy.logerr(f"Graph data not found: {graph_file}")
                return False
            
            return True

        except Exception as e:
            rospy.logerr(f"Error loading files: {e}")
            return False

    def select_missions(self, count=None, shuffle=False):
        """모든 미션을 순차 실행. count 지정 시 앞에서부터 제한. shuffle=True면 셔플."""
        missions = (self.missions_data or {}).get("missions", [])
        if shuffle:
            random.shuffle(missions)
        return missions if count is None else missions[:count]

    def world_to_grid(self, x, y):
        """월드 좌표를 그래프 그리드 좌표로 변환"""
        if not self.graph_data: return None
        res = self.graph_data['grid_resolution']
        ox, oy = self.graph_data['grid_origin_x'], self.graph_data['grid_origin_y']
        return int((x - ox) / res), int((y - oy) / res)

    def grid_to_world(self, gx, gy):
        """그리드 좌표를 월드 좌표로 변환"""
        if not self.graph_data: return None
        res = self.graph_data['grid_resolution']
        ox, oy = self.graph_data['grid_origin_x'], self.graph_data['grid_origin_y']
        return ox + gx * res, oy + gy * res

    def find_path_using_graph(self, start_pos, goal_pos):
        """GraphBuilder가 생성한 그래프를 사용하여 A* 경로 탐색"""
        if not self.graph_data:
            rospy.logerr("Graph data not available")
            return None

        # 시작점과 목표점을 그리드 좌표로 변환
        start_grid = self.world_to_grid(start_pos[0], start_pos[1])
        goal_grid = self.world_to_grid(goal_pos[0], goal_pos[1])

        if not start_grid or not goal_grid:
            rospy.logerr("Failed to convert world coordinates to grid")
            return None

        # 그래프에서 가장 가까운 노드 찾기
        graph = self.graph_data['traversable_graph']

        def find_nearest_node(target_grid):
            min_dist = float('inf')
            nearest_node = None
            for node_str in graph.keys():
                # "(x, y)" 형태의 문자열 파싱 (신뢰된 내부 포맷)
                node_coords = eval(node_str)
                dist = ((node_coords[0] - target_grid[0])**2 + (node_coords[1] - target_grid[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node_coords
            return nearest_node, min_dist

        start_node, start_dist = find_nearest_node(start_grid)
        goal_node, goal_dist = find_nearest_node(goal_grid)

        if start_dist > 50 or goal_dist > 50:
            rospy.logerr(f"Start or goal too far from graph. Start dist: {start_dist:.1f}, Goal dist: {goal_dist:.1f}")
            return None

        rospy.loginfo(f"Path planning: {start_node} -> {goal_node}")

        # A* 알고리즘 (4-연결 가정)
        from heapq import heappush, heappop

        def heuristic(a, b):
            # 맨해튼 휴리스틱
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heappush(open_set, (0, start_node))
        came_from, g_score = {}, {start_node: 0}
        f_score = {start_node: heuristic(start_node, goal_node)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal_node:
                # 경로 재구성
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                path.reverse()
                rospy.loginfo(f"Path found with {len(path)} nodes")
                return path

            current_str = f"({current[0]}, {current[1]})"
            if current_str in graph:
                for neighbor_str in graph[current_str]:
                    neighbor = eval(neighbor_str)
                    tentative_g = g_score[current] + 1  # 4-연결 비용 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal_node)
                        heappush(open_set, (f_score[neighbor], neighbor))

        rospy.logerr("No path found")
        return None

    def downsample_grid_path(self, grid_path, min_step_m=None):
        min_step_m = min_step_m or self.min_wp_spacing
        if not grid_path: return []
        waypoints, last_wp = [], None
        for gx, gy in grid_path:
            wx, wy = self.grid_to_world(gx, gy)
            if wx is None: continue
            if last_wp is None or math.hypot(wx - last_wp[0], wy - last_wp[1]) >= min_step_m:
                waypoints.append((wx, wy))
                last_wp = (wx, wy)
        gx, gy = grid_path[-1]
        w_end = self.grid_to_world(gx, gy)
        if w_end and (not waypoints or math.hypot(w_end[0]-waypoints[-1][0], w_end[1]-waypoints[-1][1]) > 1e-3):
            waypoints.append(w_end)
        return waypoints

    def prune_from_current(self, waypoints, min_start_dist=None):
        min_start_dist = min_start_dist or self.min_start_wp_dist
        cx, cy = self.vehicle_x, self.vehicle_y
        pruned = list(waypoints)
        while pruned and math.hypot(pruned[0][0]-cx, pruned[0][1]-cy) < min_start_dist:
            pruned.pop(0)
        return pruned

    def compute_actions_from_grid_path(self, grid_path):
        acts = []
        if not grid_path or len(grid_path) < 2: return acts
        res = float(self.graph_data['grid_resolution'])
        for i in range(1, len(grid_path)):
            (x0, y0), (x1, y1) = grid_path[i-1], grid_path[i]
            dx, dy = x1 - x0, y1 - y0
            action = None
            if (dx, dy) == (0, -1): action = ACTION_NORTH
            elif (dx, dy) == (1, 0): action = ACTION_EAST
            elif (dx, dy) == (0, 1): action = ACTION_SOUTH
            elif (dx, dy) == (-1, 0): action = ACTION_WEST
            label = ACTION_MAP.get(action, "UNKNOWN")
            planned_dist = res * math.hypot(dx, dy)
            acts.append({"from_grid": [x0, y0], "to_grid": [x1, y1], "action": action, "action_label": label, "planned_distance_m": planned_dist})
        return acts

    def compress_actions(self, actions):
        if not actions: return []
        runs, cur_dir, steps = [], actions[0]["action_label"], 1
        res = float(self.graph_data['grid_resolution'])
        for i in range(1, len(actions)):
            d = actions[i]["action_label"]
            if d != cur_dir or d == "UNKNOWN":
                runs.append({"dir": cur_dir, "steps": int(steps), "meters": float(steps * res)})
                cur_dir, steps = d, 1
            else:
                steps += 1
        runs.append({"dir": cur_dir, "steps": int(steps), "meters": float(steps * res)})
        return runs
    def _extract_label_and_xy(self, wp):
        # 라벨: A or B or label
        label = wp.get("A") or wp.get("B") or wp.get("label", "unknown")
        pos = wp.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            raise ValueError(f"Bad waypoint format: {wp}")
        # z는 무시
        return label, (float(pos[0]), float(pos[1]))

    def execute_graph_path(self, path, target_label, save_folder):
        """[수정] 경로 실행 + (성공여부, 세그먼트 로그) 반환"""
        # [수정] phase_meta를 함수 내부에서 생성
        phase_meta = {}

        if not path or len(path) < 2:
            phase_meta["result"] = "no_path"
            return False, phase_meta

        # 메타: 계획 경로(그리드/월드)
        phase_meta.setdefault("planned_grid_path", path)
        world_waypoints = []
        for gx, gy in path:
            w = self.grid_to_world(gx, gy)
            if w:
                world_waypoints.append([float(w[0]), float(w[1])])
        phase_meta.setdefault("planned_world_path", world_waypoints)

        # 액션 시퀀스(그리드 기준)
        phase_meta.setdefault("grid_actions", self.compute_actions_from_grid_path(path))

        # 👇 압축된 방향-스텝(run-length) 추가 저장: (North, 3) 같은 요약
        phase_meta.setdefault("compressed_actions", self.compress_actions(phase_meta["grid_actions"]))

        # 실제 주행은 다운샘플 웨이포인트 기준으로 세그먼트화
        waypoints = self.downsample_grid_path(path)
        waypoints = self.prune_from_current(waypoints, self.min_start_wp_dist)
        if not waypoints:
            rospy.loginfo("All waypoints too close to start; skipping path.")
            phase_meta["result"] = "skipped_too_close"
            return True, phase_meta

        rospy.loginfo(f"Executing downsampled path with {len(waypoints)} waypoints to {target_label}")
        rate = rospy.Rate(10)

        i = 0
        phase_meta.setdefault("segments", [])
        while i < len(waypoints) and not rospy.is_shutdown():
            tx, ty = waypoints[i]
            ok, seg = self.move_to_position(tx, ty, rate)

            # 센서 저장
            prefix = f"step_{i:03d}_x_{tx:.2f}_y_{ty:.2f}"
            #self.save_data_to_folder(prefix, save_folder)
            capture_meta = self.save_data_to_folder(prefix, save_folder, target_world_xy=(tx, ty))
            if capture_meta:
                phase_meta.setdefault("captures", []).append(capture_meta)
            # 세그먼트 로그
            seg_entry = {
                "segment_idx": i,
                "target_waypoint": [float(tx), float(ty)],
                "ok": bool(ok),
                "actual_distance_m": seg["actual_distance_m"],
                "duration_s": seg["duration_s"],
                "trajectory": seg["trajectory"],  # 용량 크면 이후 간헐 샘플링 고려
            }
            phase_meta["segments"].append(seg_entry)

            # 실패 처리
            if not ok:
                dist = math.hypot(tx - self.vehicle_x, ty - self.vehicle_y)
                if dist < 0.45:
                    rospy.logwarn(f"Near waypoint but stuck (dist={dist:.2f}). Treat as reached.")
                    i += 1
                    continue
                rospy.logwarn(f"Skip unreachable waypoint ({tx:.2f}, {ty:.2f}) and continue.")
                i += 1
                continue

            # 정상 도달
            rospy.sleep(0.3)
            i += 1

        rospy.loginfo(f"Finished path to {target_label}")
        # [수정] 성공 여부와 함께 채워진 phase_meta를 반환
        phase_meta["result"] = "success" if i == len(waypoints) else "partial_success"
        return phase_meta["result"] == "success", phase_meta

    def move_to_position(self, target_x, target_y, rate):
        """웨이포인트까지 이동. (성공여부, {실제거리,시간,궤적}) 반환"""
        timeout = rospy.Time.now() + rospy.Duration(45.0)
        prev_dist = None
        last_progress_time = rospy.Time.now()

        # 주행 재개
        self.resume_pub.publish(Empty())
        self.resume_goal_pub.publish(Empty())

        carrot_boost = 1.0

        # 로그용
        t0 = rospy.Time.now()
        last_x, last_y = self.vehicle_x, self.vehicle_y
        actual_dist = 0.0
        traj = []  # [{t,x,y,yaw}...]

        while not rospy.is_shutdown() and rospy.Time.now() < timeout:
            cx, cy = self.vehicle_x, self.vehicle_y
            dx, dy = (target_x - cx), (target_y - cy)
            dist = math.hypot(dx, dy)

            # 궤적/거리 적분
            yaw = self.vehicle_yaw
            traj.append({
                "t": (rospy.Time.now() - t0).to_sec(),
                "x": float(cx), "y": float(cy), "yaw": float(yaw)
            })
            step = math.hypot(cx - last_x, cy - last_y)
            actual_dist += step
            last_x, last_y = cx, cy

            # carrot 계산
            base = self.carrot_min
            carrot_len = max(self.carrot_min, min(self.carrot_max, base * carrot_boost))
            if dist > carrot_len:
                scale = carrot_len / dist
                carrot_x = cx + dx * scale
                carrot_y = cy + dy * scale
            else:
                carrot_x, carrot_y = target_x, target_y

            # 헤딩: 멀면 목표방향, 가까우면 현재 유지
            heading_to_target = math.atan2(dy, dx)
            theta_cmd = heading_to_target if dist > 0.6 else self.vehicle_yaw
            self.waypoint_pub.publish(Pose2D(x=carrot_x, y=carrot_y, theta=theta_cmd))

            # 도달 판정
            if dist < self.waypoint_reach_dis:
                dur = (rospy.Time.now() - t0).to_sec()
                rospy.loginfo(f"Reached target. Distance: {dist:.3f}")
                return True, {"actual_distance_m": float(actual_dist), "duration_s": float(dur), "trajectory": traj}

            # 진행 모니터링
            if prev_dist is None or dist + 1e-3 < prev_dist:
                prev_dist = dist
                last_progress_time = rospy.Time.now()
                carrot_boost = 1.0
            else:
                idle = (rospy.Time.now() - last_progress_time).to_sec()
                if idle > 4.0:
                    rospy.logwarn("No progress >4s: increase carrot and continue")
                    carrot_boost = min(1.6, carrot_boost * 1.2)
                if idle > 7.0:
                    rospy.logwarn("No progress >7s: marking as failed for this waypoint")
                    dur = (rospy.Time.now() - t0).to_sec()
                    return False, {"actual_distance_m": float(actual_dist), "duration_s": float(dur), "trajectory": traj}
                if idle > 2.0:
                    now = rospy.Time.now()
                    if not hasattr(self, "_last_resume") or (now - self._last_resume).to_sec() > 2.5:
                        self.resume_pub.publish(Empty())
                        self.resume_goal_pub.publish(Empty())
                        self._last_resume = now

            rate.sleep()

        rospy.logwarn(f"Timeout reaching target ({target_x:.2f}, {target_y:.2f})")
        dur = (rospy.Time.now() - t0).to_sec()
        return False, {"actual_distance_m": float(actual_dist), "duration_s": float(dur), "trajectory": traj}

    def save_data_to_folder(self, prefix, folder_path, target_world_xy):
        """이미지/라이다 저장 + 캡처 메타 반환"""
        if self.latest_image_msg is None or self.latest_lidar_msg is None:
            rospy.logwarn(f"No sensor data available for {prefix}")
            return None

        try:
            # 파일 경로
            image_path = os.path.join(folder_path, f"{prefix}_image.jpg")
            lidar_path = os.path.join(folder_path, f"{prefix}_lidar.npy")

            # 이미지 저장
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image_msg, "bgr8")
            cv2.imwrite(image_path, cv_image)

            # 라이다 저장
            lidar_array = self.pointcloud2_to_array(self.latest_lidar_msg)
            np.save(lidar_path, lidar_array)

            # 캡처 시점 로봇 포즈/그리드
            rx, ry, ryaw = float(self.vehicle_x), float(self.vehicle_y), float(self.vehicle_yaw)
            rgi, rgj = self.world_to_grid(rx, ry)  # (grid_x, grid_y)
            tx, ty = float(target_world_xy[0]), float(target_world_xy[1])
            tgi, tgj = self.world_to_grid(tx, ty)

            # 센서 타임스탬프(가능할 때)
            img_stamp = float(self.latest_image_msg.header.stamp.to_sec()) if hasattr(self.latest_image_msg, "header") else None
            lidar_stamp = float(self.latest_lidar_msg.header.stamp.to_sec()) if hasattr(self.latest_lidar_msg, "header") else None

            capture_meta = {
                "filename_image": os.path.basename(image_path),
                "filename_lidar": os.path.basename(lidar_path),
                "wall_time_iso": datetime.now().isoformat(),
                "ros_time": float(rospy.Time.now().to_sec()),
                "target_world_xy": [tx, ty],
                "target_grid_ij": [int(tgi), int(tgj)],
                "robot_world_xy": [rx, ry],
                "robot_yaw": ryaw,
                "robot_grid_ij": [int(rgi), int(rgj)],
                "image_stamp": img_stamp,
                "lidar_stamp": lidar_stamp,
            }
            return capture_meta

        except Exception as e:
            rospy.logerr(f"Failed to save data for {prefix}: {e}")
            return None

    def pointcloud2_to_array(self, cloud_msg):
        """PointCloud2 메시지를 numpy array로 변환"""
        points = []
        for point in pc2.read_points(cloud_msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def wait_for_graph_ready(self, timeout_sec=60):
        """GraphBuilder가 그래프를 발행/저장할 때까지 대기"""
        try:
            rospy.wait_for_message('/exploration/traversable_graph', OccupancyGrid, timeout=timeout_sec/2.0)
        except rospy.ROSException:
            rospy.logwarn("No /exploration/traversable_graph within timeout; falling back to file polling")

        graph_file = "/tmp/graph_data.json"
        t0 = time.time()
        while (time.time() - t0) < timeout_sec and not rospy.is_shutdown():
            if os.path.exists(graph_file) and os.path.getsize(graph_file) > 0:
                try:
                    with open(graph_file, 'r') as f:
                        self.graph_data = json.load(f)
                    if self.graph_data.get('traversable_graph'):
                        rospy.loginfo("Graph data ready")
                        return True
                except Exception as e:
                    rospy.logwarn(f"Graph file present but not readable yet: {e}")
            time.sleep(1.0)
        return False


    def run_mission(self):
        """[수정] 다중 경유지 미션 실행 함수"""
        # [수정] /current_scene_name 파라미터가 설정될 때까지 대기
        rospy.loginfo("Waiting for /current_scene_name parameter from the simulator...")
        scene_name_param = "/current_scene_name"
        wait_start_time = rospy.Time.now()
        while not rospy.has_param(scene_name_param) and (rospy.Time.now() - wait_start_time).to_sec() < 30.0:
            if rospy.is_shutdown(): return
            rospy.sleep(0.5)

        if not rospy.has_param(scene_name_param):
            rospy.logerr("Timed out waiting for /current_scene_name parameter. Is the Unity simulator running?")
            return
        
        # [수정] 파라미터 확인 후 파일 로드
        self.load_missions_and_graph()

        if not self.missions_data or not self.graph_data:
            rospy.logerr("Required data not loaded. Exiting.")
            return

        selected_missions = self.select_missions(count=None, shuffle=False)
        if not selected_missions:
            rospy.logerr("No missions found.")
            return

        rospy.loginfo("Starting graph-based missions with multiple waypoints...")

        try:
            first_odom = rospy.wait_for_message("/state_estimation", Odometry, timeout=10)
            self.pose_handler(first_odom)
        except rospy.ROSException:
            rospy.logerr("No odometry message received. Exiting.")
            return

        rospy.loginfo("Waiting for navigation system to be ready...")
        rospy.sleep(3.0)
        self.resume_pub.publish(Empty())
        self.resume_goal_pub.publish(Empty())
        rospy.loginfo("Resume navigation signal sent.")
        rospy.sleep(1.0)

        for mission_idx, mission in enumerate(selected_missions):
            rospy.loginfo(f"\n=== Starting Mission {mission_idx + 1} ===")
            rospy.loginfo(f"Instruction: {mission['instruction']}")

            waypoints = mission.get("waypoints", [])
            if not waypoints:
                rospy.logwarn("Mission has no waypoints. Skipping.")
                continue

            # [수정] scene 이름을 포함하여 저장 경로 생성
            first_wp_label, _ = self._extract_label_and_xy(waypoints[0])
            last_wp_label,  _ = self._extract_label_and_xy(waypoints[-1])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_stub = "".join(c for c in f"{mission_idx+1:03d}_{first_wp_label}_{last_wp_label}" if c.isalnum() or c in ('_', '-'))
            
            scene_save_path = os.path.join(self.save_base_path, self.current_scene)
            mission_folder = os.path.join(scene_save_path, f"{folder_stub}_{ts}")
            
            os.makedirs(mission_folder, exist_ok=True)
            rospy.loginfo(f"Data will be saved to: {mission_folder}")

            # [수정] 메타데이터에 scene_name 추가
            mission_meta = {
                "mission_id": f"mission_{mission_idx + 1}",
                "instruction": mission["instruction"],
                "waypoints": waypoints, # 전체 웨이포인트 정보 저장
                "folder_name": os.path.basename(mission_folder),
                "scene_name": self.current_scene,
                "graph_meta": {
                    "grid_resolution": self.graph_data['grid_resolution'],
                    "grid_origin_x": self.graph_data['grid_origin_x'],
                    "grid_origin_y": self.graph_data['grid_origin_y'],
                },
                "created_at": datetime.now().isoformat(),
                "phases": [] # 각 경로 단계를 저장할 리스트
            }

            # 경유지 순차 실행
            mission_success = True
            for i, waypoint_info in enumerate(waypoints):
                label, goal_pos = self._extract_label_and_xy(waypoint_info)
                rospy.loginfo(f"--- Phase {i}: Moving to {label} ---")

                current_pos = (self.vehicle_x, self.vehicle_y)
                path_to_goal = self.find_path_using_graph(current_pos, goal_pos)

                if path_to_goal:
                    ok, phase_meta = self.execute_graph_path(path_to_goal, label, mission_folder)
                    if not ok:
                        rospy.logerr(f"Failed to reach waypoint {label} for mission {mission_idx + 1}")
                        mission_success = False
                else:
                    rospy.logerr(f"No path to {label} for mission {mission_idx + 1}")
                    phase_meta = {"planned_grid_path": None, "result": "no_path_found"}
                    mission_success = False

                mission_meta["phases"].append(phase_meta)


            mission_meta["success"] = mission_success
            
            # 최종 메타데이터 저장
            metadata_path = os.path.join(mission_folder, "mission_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(mission_meta, f, indent=2)

            rospy.sleep(2.0)

        rospy.loginfo("All missions completed!")
        rospy.signal_shutdown("missions done")
        return

# [수정] 스크립트 실행을 위한 엔트리포인트 추가
if __name__ == "__main__":
    try:
        node = MissionNode()
        node.run_mission()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("MissionNode crashed:\n" + traceback.format_exc())
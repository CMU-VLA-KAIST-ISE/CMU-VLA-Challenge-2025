#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, time
from ast import literal_eval
from datetime import datetime
from heapq import heappush, heappop

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros, tf2_sensor_msgs
import tf.transformations as tft
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty


class GraphExplorer:
    """
    - graph + target_sets 로드(ready까지 대기)
    - explore_plan: 'edge', 'edge+skeleton', 'edge+skeleton+interior'
    - budget_max_targets: 총 방문 상한(0이면 무제한)
    - A* (clearance cost) + LOS shortcut + carrot follower
    """

    def __init__(self):
        rospy.init_node("graph_explorer", anonymous=True)

        # --- params ---
        self.graph_file = rospy.get_param("~graph_file", "/tmp/graph_data.json")
        self.save_dir = rospy.get_param("~save_dir", "/home/kwakjs/CMU-VLA-Challenge/collected_data_cs")
        self.random_start = rospy.get_param("~random_start", False)
        self.wait_graph_timeout = float(rospy.get_param("~wait_graph_timeout", 90.0))
        self.min_wp_spacing = float(rospy.get_param("~min_wp_spacing", 0.6))
        self.min_start_wp_dist = float(rospy.get_param("~min_start_wp_dist", 0.3))
        self.reach_dist = float(rospy.get_param("~reach_dist", 0.30))
        self.carrot_min = float(rospy.get_param("~carrot_min", 1.0))
        self.carrot_max = float(rospy.get_param("~carrot_max", 2.0))

        # NEW: 탐색 플랜 & 예산
        self.explore_plan = rospy.get_param("~explore_plan", "edge+skeleton")
        self.budget_max_targets = int(rospy.get_param("~budget_max_targets", 0))  # 0 -> unlimited
        self.shuffle_targets = bool(rospy.get_param("~shuffle_targets", False))    # True면 좀 랜덤

        # topics/frames
        self.odom_topic = rospy.get_param("~odom_topic", "/state_estimation")
        # [수정] C++ 노드가 발행하는 최종 토픽 이름으로 변경
        self.lidar_topic = rospy.get_param("~lidar_topic", "/color_scan")
        self.wp_topic = rospy.get_param("~wp_topic", "/way_point_with_heading")
        self.target_frame = rospy.get_param("~target_frame", "map")

        self.target_min_gap_m = float(rospy.get_param("~target_min_gap_m", 0.9))
        self.order_policy     = rospy.get_param("~order_policy", "boustrophedon")  # 'boustrophedon' | 'nn'
        self.order_cell_m     = float(rospy.get_param("~order_cell_m", 1.2))
        self.visit_policy     = rospy.get_param("~visit_policy", "ordered")        # 'ordered' | 'greedy'

        # dirs
        manual_scene = rospy.get_param("~scene_name", "")
        scene = manual_scene or rospy.get_param("/current_scene_name", "unknown_scene_fallback")
        run_tag = rospy.get_param("~run_tag", "")
        use_ts = rospy.get_param("~use_timestamp", True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dname = f"{run_tag+'_' if run_tag else ''}{ts if use_ts else 'explore_run'}"
        self.out_dir = os.path.join(self.save_dir, scene, dname)
        os.makedirs(self.out_dir, exist_ok=True)
        rospy.loginfo(f"[GraphExplorer] save: {self.out_dir}")

        # ROS I/O
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.cb_lidar, queue_size=5)
        self.pub_wp = rospy.Publisher(self.wp_topic, Pose2D, queue_size=1)
        self.pub_resume = rospy.Publisher("/resume", Empty, queue_size=1, latch=True)
        self.pub_resume_goal = rospy.Publisher("/resume_goal", Empty, queue_size=1, latch=True)

        # TF
        self.tf_buffer = tf2_ros.Buffer(); self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # state
        self.x=self.y=self.yaw=0.0
        self.latest_lidar=None
        self.points_world_chunks=[]; self.chunk_points=[]

        # map
        self.graph={}
        self.res=0.1; self.ox=0.0; self.oy=0.0
        self.free_mask=None; self.clearance=None

        # targets
        self.target_sets={"edge":[], "skeleton":[], "interior":[]}
        self.targets=[]

        self.load_graph_and_targets()
        self.load_free_mask()
        self.compose_targets()
        self.control_mode = rospy.get_param("~control_mode", "carrot")  # 'carrot' | 'fixed_wp'
        self.wp_timeout = float(rospy.get_param("~wp_timeout", 12.0))
        self.progress_epsilon = float(rospy.get_param("~progress_epsilon", 0.05))


    # ---------- load ----------
    def load_graph_and_targets(self):
        t0=time.time()
        while not rospy.is_shutdown() and (time.time()-t0) < self.wait_graph_timeout:
            if os.path.exists(self.graph_file) and os.path.getsize(self.graph_file)>0:
                try:
                    with open(self.graph_file,"r") as f: gd=json.load(f)
                    g=gd.get("traversable_graph",{})
                    self.graph={literal_eval(k):[literal_eval(s) for s in v] for k,v in g.items()}
                    self.res=float(gd["grid_resolution"]); self.ox=float(gd["grid_origin_x"]); self.oy=float(gd["grid_origin_y"])
                    ts=gd.get("target_sets",{})
                    for k in ("edge","skeleton","interior"):
                        self.target_sets[k]=[literal_eval(s) for s in ts.get(k,[])]
                    ready=bool(gd.get("ready",False))
                    if self.graph and ready:
                        rospy.loginfo(f"[GraphExplorer] loaded graph={len(self.graph)} "
                                      f"edge={len(self.target_sets['edge'])} "
                                      f"skel={len(self.target_sets['skeleton'])} "
                                      f"int={len(self.target_sets['interior'])}")
                        return
                    else:
                        rospy.logwarn_throttle(1.0,"[GraphExplorer] waiting graph ready…")
                except Exception as e:
                    rospy.logwarn_throttle(1.0,f"[GraphExplorer] parse error: {e}")
            rospy.sleep(0.5)
        if not self.graph:
            rospy.logerr("[GraphExplorer] graph file not ready"); raise SystemExit(1)

    def load_free_mask(self):
        p="/tmp/free_mask.npz"
        if os.path.exists(p):
            try:
                dat=np.load(p)
                self.free_mask=dat["free"].astype(np.uint8)
                dt=cv2.distanceTransform(self.free_mask, cv2.DIST_L2, 3)
                self.clearance=dt*float(dat["res"])
                self.res=float(dat["res"]); self.ox=float(dat["ox"]); self.oy=float(dat["oy"])
                rospy.loginfo("[GraphExplorer] loaded free_mask/clearance")
            except Exception as e:
                rospy.logwarn(f"[GraphExplorer] load mask fail: {e}")

    # ---------- transforms/planner ----------
    def grid_to_world(self, ij): return (self.ox+(ij[0]+0.5)*self.res, self.oy+(ij[1]+0.5)*self.res)
    def world_to_grid(self, xy): return (int(math.floor((xy[0]-self.ox)/self.res)), int(math.floor((xy[1]-self.oy)/self.res)))

    def step_cost(self,u,v):
        base=1.0 if (u[0]==v[0] or u[1]==v[1]) else math.sqrt(2)
        if self.clearance is None: return base
        c=self.clearance[v[1],v[0]]
        return base*(1.0+2.5*max(0.0,(0.45-c)))

    def astar(self,start,goal):
        if start==goal: return [start]
        def h(a,b):
            dx,dy=abs(a[0]-b[0]),abs(a[1]-b[1])
            return (dx+dy)+(math.sqrt(2)-2)*min(dx,dy)
        openq=[]; heappush(openq,(0.0,start)); g={start:0.0}; parent={}; closed=set()
        while openq:
            _,u=heappop(openq)
            if u in closed: continue
            closed.add(u)
            if u==goal:
                p=[u]
                while u in parent: u=parent[u]; p.append(u)
                return list(reversed(p))
            for v in self.graph.get(u,[]):
                ng=g[u]+self.step_cost(u,v)
                if v not in g or ng<g[v]:
                    g[v]=ng; parent[v]=u; heappush(openq,(ng+h(v,goal),v))
        return None

    def has_line_of_sight(self,a,b):
        if self.free_mask is None: return False
        x0,y0=a; x1,y1=b
        dx=abs(x1-x0); dy=abs(y1-y0); sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1; err=dx-dy; x,y=x0,y0
        while True:
            if self.free_mask[y,x]==0: return False
            if x==x1 and y==y1: break
            e2=2*err
            if e2>-dy: err-=dy; x+=sx
            if e2< dx: err+=dx; y+=sy
        return True

    def smooth_path(self,p):
        if not p or len(p)<=2 or self.free_mask is None: return p
        out=[p[0]]; i=0
        while i<len(p)-1:
            j=len(p)-1
            while j>i+1 and not self.has_line_of_sight(p[i],p[j]): j-=1
            out.append(p[j]); i=j
        return out

    def downsample_grid_path(self, path):
        path=self.smooth_path(path)
        if not path: return []
        wps=[]; last=None; last_dir=None 
        def add(k):
            wx,wy=self.grid_to_world(path[k]); wps.append((wx,wy)); return (wx,wy)
        for k in range(1,len(path)):
            dx=path[k][0]-path[k-1][0]; dy=path[k][1]-path[k-1][1]
            d=(dx,dy)
            if last_dir is not None and d!=last_dir: last=add(k-1)
            wx,wy=self.grid_to_world(path[k])
            if last is None or math.hypot(wx-last[0],wy-last[1])>=self.min_wp_spacing: last=add(k)
            last_dir=d
        goal=self.grid_to_world(path[-1])
        if not wps or wps[-1]!=goal: wps.append(goal)
        if len(wps)==1: wps=[wps[0],goal]
        if len(wps) == 2:
        # 출발점 전방으로 0.8m, 목표 쪽으로 좌/우 0.6m 오프셋 하나 추가
            sx, sy = wps[0]; gx, gy = wps[-1]
            hdg = math.atan2(gy - sy, gx - sx)
            mid = (sx + 0.8*math.cos(hdg), sy + 0.8*math.sin(hdg))
            # 왼쪽으로 살짝 벌려 회전 공간 확보
            mid = (mid[0] - 0.6*math.sin(hdg), mid[1] + 0.6*math.cos(hdg))
            wps.insert(1, mid)
        return wps

    # ---------- controller ----------
    def follow_path_with_carrot(self, wps):
        if not wps:
            return True

        rate = rospy.Rate(10)
        last_progress_time = rospy.Time.now()
        carrot_boost = 1.0

        self.pub_resume.publish(Empty()); self.pub_resume_goal.publish(Empty())

        last_pose = (self.x, self.y)
        last_goal_dist = math.hypot(wps[-1][0] - self.x, wps[-1][1] - self.y)
        last_closest_idx = 0
        last_theta = None  # heading hysteresis

        def closest_index_forward(cx, cy, start_idx):
            best_i = start_idx
            best_d2 = float("inf")
            for i in range(start_idx, len(wps)):
                dx = wps[i][0] - cx; dy = wps[i][1] - cy
                d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2 = d2; best_i = i
            return best_i

        def advance_carrot(i_closest, cx, cy, yaw):
            Lmin, Lmax = self.carrot_min, self.carrot_max
            L = max(Lmin, min(Lmax, Lmin * carrot_boost))

            hx, hy = math.cos(yaw), math.sin(yaw)  # 로봇 전방 단위벡터

            acc = 0.0
            j = max(0, min(i_closest, len(wps)-2))
            while j+1 < len(wps):
                segx = wps[j+1][0]-wps[j][0]
                segy = wps[j+1][1]-wps[j][1]
                seg  = max(1e-6, math.hypot(segx, segy))

                if acc + seg >= L:
                    r  = (L - acc)/seg
                    wx = wps[j][0] + r*segx
                    wy = wps[j][1] + r*segy

                    # ★ 전방 반평면 강제: carrot이 뒤라면 세그먼트 하나 더 앞으로
                    if (wx - cx)*hx + (wy - cy)*hy < 0.0 and j+1 < len(wps)-1:
                        j += 1
                        continue
                    return wx, wy, j

                acc += seg
                j += 1

            return wps[-1][0], wps[-1][1], len(wps)-2

        def angle_wrap(a):
            while a > math.pi:  a -= 2*math.pi
            while a < -math.pi: a += 2*math.pi
            return a

        # 시작이 거의 골이면 즉시 성공
        if math.hypot(self.x - wps[-1][0], self.y - wps[-1][1]) < self.reach_dist:
            return True

        seg_start = rospy.Time.now()
        last_yaw = self.yaw
        while not rospy.is_shutdown():
            cx, cy, yaw = self.x, self.y, self.yaw
            self.capture_lidar_once()

            i_closest = closest_index_forward(cx, cy, last_closest_idx)
            last_closest_idx = max(last_closest_idx, i_closest)

            wx, wy, seg_idx = advance_carrot(i_closest, cx, cy, yaw)

            # ★ 경로 접선 방향으로 헤딩 생성 (bearing보다 안정적)
            if seg_idx < len(wps)-1:
                tx = wps[seg_idx+1][0] - wps[seg_idx][0]
                ty = wps[seg_idx+1][1] - wps[seg_idx][1]
                theta_cmd = math.atan2(ty, tx)
            else:
                theta_cmd = math.atan2(wy - cy, wx - cx)

            # ★ 헤딩 히스테리시스: 큰 급변(>90°)은 완만하게 블렌딩
            if last_theta is not None:
                d = angle_wrap(theta_cmd - last_theta)
                limit = math.radians(60.0)
                if abs(d) > limit:
                    theta_cmd = last_theta + max(-limit, min(limit, d))
            last_theta = theta_cmd

            self.pub_wp.publish(Pose2D(x=wx, y=wy, theta=theta_cmd))

            gdist = math.hypot(wps[-1][0] - cx, wps[-1][1] - cy)
            if gdist < self.reach_dist:
                return True

            moved = math.hypot(cx - last_pose[0], cy - last_pose[1])
            goal_improved = (last_goal_dist - gdist) > 0.02
            
            yaw_changed = abs(angle_wrap(self.yaw - last_yaw)) > math.radians(7)
            if moved > 0.02 or goal_improved or yaw_changed:
                last_progress_time = rospy.Time.now()
                carrot_boost = 1.0
                last_pose = (cx, cy)
                last_goal_dist = gdist
                last_yaw = self.yaw
            else:
                idle = (rospy.Time.now() - last_progress_time).to_sec()
                if idle > 4.0:
                    rospy.logwarn_throttle(1.0, "No progress >4s: increasing carrot")
                    carrot_boost = min(1.8, carrot_boost * 1.15)
                if idle > 12.0:
                    rospy.logwarn("No progress >12s: backing up and replan")
                    self.pub_wp.publish(Pose2D(x=cx-0.25*math.cos(yaw), y=cy-0.25*math.sin(yaw), theta=yaw))
                    rospy.sleep(1.0)
                    return False
            if (rospy.Time.now() - seg_start).to_sec() > 90.0:
                rospy.logwarn("Segment timeout >90s; aborting this segment")
                return False

            rate.sleep()
        return False


    def execute_path_world(self, wps, max_replans=2):
        if not wps: return True
        for t in range(max_replans + 1):
            ok = (self.follow_path_with_carrot(wps) if self.control_mode == "fixed_wp"
                else self.drive_fixed_waypoints(wps))
            if ok:
                return True
            # 실패 시 세그먼트 재계획
            cur = self.nearest_node_world((self.x, self.y))
            tgt = self.world_to_grid(wps[-1])
            p2 = self.astar(cur, tgt)
            if not p2:
                break
            wps = self.downsample_grid_path(p2)
            rospy.loginfo(f"[GraphExplorer] Replan #{t+1}: {len(wps)} waypoints")
        return False

    # ---------- lidar merge ----------
    def cb_odom(self, msg: Odometry):
        self.x=msg.pose.pose.position.x; self.y=msg.pose.pose.position.y
        q=msg.pose.pose.orientation; _,_,self.yaw=tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
    def cb_lidar(self, msg: PointCloud2): self.latest_lidar=msg
    
    def capture_lidar_once(self):
        if self.latest_lidar is None:
            rospy.logwarn_throttle(5.0, "[Capture] No LiDAR data received yet.")
            return
        try:
            # [수정] do_transform_cloud 대신, 변환 행렬(matrix)만 가져옵니다.
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, 
                self.latest_lidar.header.frame_id,
                self.latest_lidar.header.stamp, 
                rospy.Duration(0.2)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"[Capture] TF transform failed: {e}")
            return
        
        # 변환 행렬 생성
        trans = transform.transform.translation
        rot = transform.transform.rotation
        transform_matrix = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        transform_matrix[0, 3] = trans.x
        transform_matrix[1, 3] = trans.y
        transform_matrix[2, 3] = trans.z

        pts=[]
        # [수정] 변환 전의 원본 데이터(self.latest_lidar)에서 'rgb' 필드를 읽습니다.
        for p in pc2.read_points(self.latest_lidar, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb_float = p[0], p[1], p[2], p[3]
            
            # 수동으로 좌표 변환 수행
            point_in = np.array([x, y, z, 1.0])
            point_out = np.dot(transform_matrix, point_in)
            
            # 색상 정보 분리
            rgb_uint = np.array([rgb_float], dtype=np.float32).view(np.uint32)[0]
            r = (rgb_uint >> 16) & 0xFF
            g = (rgb_uint >> 8) & 0xFF
            b = rgb_uint & 0xFF

            # 변환된 좌표와 원본 색상 정보를 함께 저장
            pts.append((point_out[0], point_out[1], point_out[2], r, g, b))
            if len(pts) >= 5000: break
        
        if pts:
            self.chunk_points.extend(pts)
            rospy.loginfo_throttle(2.0, f"[Capture] Collected {len(pts)} points. Total in chunk: {len(self.chunk_points)}")
            if len(self.chunk_points) >= 20000:
                self.flush_chunk()

    def flush_chunk(self):
        if not self.chunk_points: return
        arr=np.asarray(self.chunk_points,dtype=np.float32)
        self.points_world_chunks.append(arr); self.chunk_points=[]
    def merge_and_save(self):
        self.flush_chunk()
        merged=np.vstack(self.points_world_chunks) if self.points_world_chunks else np.empty((0,6),dtype=np.float32)
        np.save(os.path.join(self.out_dir,"merged_lidar_world_color.npy"), merged)
        with open(os.path.join(self.out_dir,"merged_lidar_world_color.ply"),"w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {merged.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for x,y,z,r,g,b in merged: f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
        rospy.loginfo("[GraphExplorer] saved merged LiDAR")
    

    ####
    def drive_fixed_waypoints(self, wps):
        """네 object-list 코드 느낌: 각 WP를 '도달할 때까지' 유지."""
        if not wps: return True
        rate = rospy.Rate(10)
        for (wx, wy) in wps:
            start = rospy.Time.now()
            best = 1e18
            while not rospy.is_shutdown():
                d = math.hypot(self.x - wx, self.y - wy)
                if d < self.reach_dist:
                    break
                # 목표점은 고정, 헤딩는 해당 목표점으로
                theta = math.atan2(wy - self.y, wx - self.x)
                self.pub_wp.publish(Pose2D(x=wx, y=wy, theta=theta))
                self.capture_lidar_once()

                # 진행성 체크(최단거리 갱신 없고, 타임아웃이면 리턴 False → 상위에서 리플랜)
                if d < best - self.progress_epsilon:
                    best = d
                    start = rospy.Time.now()
                elif (rospy.Time.now() - start).to_sec() > self.wp_timeout:
                    rospy.logwarn("Fixed-WP: timeout/no progress → replan")
                    return False
                rate.sleep()
        return True
    ####
    # ---------- helpers/explore ----------
    def nearest_node_world(self, xy):
        gi=self.world_to_grid(xy); best=None; bd=9e9
        for n in self.graph.keys():
            d=abs(n[0]-gi[0])+abs(n[1]-gi[1])
            if d<bd: bd=d; best=n
        return best
    def nearest_target(self, node, cand):
        bx,by=node; best=None; bd=9e9
        for n in cand:
            d=(n[0]-bx)**2+(n[1]-by)**2
            if d<bd: bd=d; best=n
        return best

    def explore(self):
        if not self.graph or not self.targets:
            rospy.logerr("[GraphExplorer] empty graph/targets"); return

        cur = self.nearest_node_world((self.x, self.y))

        if self.visit_policy.lower() == "ordered":
            # 스윕 순서를 그대로 따라감
            for i, tgt in enumerate(self.targets):
                if rospy.is_shutdown(): break
                p = self.astar(cur, tgt)
                if p is None:
                    rospy.logwarn(f"[GraphExplorer] no path to {tgt}, skip")
                    continue
                wps = self.downsample_grid_path(p)
                rospy.loginfo(f"[GraphExplorer] [{i+1}/{len(self.targets)}] move via {len(wps)} WPs")
                ok = self.execute_path_world(wps)
                cur = tgt if ok else self.nearest_node_world((self.x, self.y))
            rospy.loginfo("[GraphExplorer] exploration done (ordered)")
        else:
            # 기존 greedy
            unvisited = set(self.targets)
            while unvisited and not rospy.is_shutdown():
                tgt = self.nearest_target(cur, unvisited)
                if tgt is None: break
                p = self.astar(cur, tgt)
                if p is None:
                    rospy.logwarn(f"[GraphExplorer] no path to {tgt}, drop")
                    unvisited.discard(tgt); continue
                wps = self.downsample_grid_path(p)
                rospy.loginfo(f"[GraphExplorer] move to {tgt} via {len(wps)} waypoints "
                            f"(remain {len(unvisited)})")
                if self.execute_path_world(wps):
                    cur = tgt; unvisited.discard(tgt)
                else:
                    cur = self.nearest_node_world((self.x, self.y))
        rospy.loginfo("[GraphExplorer] exploration done (greedy)")


    def compose_targets(self):
        """타깃 합치기 -> 간격 확보(포아송 근사) -> 스윕 순서 정렬 -> 예산 컷"""
        plan = self.explore_plan.replace(" ", "").lower()
        seq = []
        if "edge" in plan:     seq += self.target_sets["edge"]
        if "skeleton" in plan: seq += self.target_sets["skeleton"]
        if "interior" in plan: seq += self.target_sets["interior"]

        # 1) 간격 확보: 그리드 버킷 기반 Poisson-disk 근사
        gap_cells = max(1, int(round(self.target_min_gap_m / self.res)))
        seen = set()
        filtered = []
        for (x, y) in seq:
            key = (x // gap_cells, y // gap_cells)
            if key in seen:
                continue
            seen.add(key)
            filtered.append((x, y))

        # 2) 스윕 순서 생성
        if self.order_policy.lower() == "boustrophedon":
            cell_h = max(1, int(round(self.order_cell_m / self.res)))
            rows = {}
            for (x, y) in filtered:
                ry = y // cell_h
                rows.setdefault(ry, []).append((x, y))
            ordered = []
            for r in sorted(rows.keys()):
                row = rows[r]
                row.sort(key=lambda p: p[0])      # x 오름차순
                if (r % 2) == 1:                  # 홀수 줄은 뒤집어서 왕복 스윕
                    row.reverse()
                ordered.extend(row)
        else:
            # fallback: 최근접 탐욕 순서(초기 순서 무시)
            ordered = []
            remain = set(filtered)
            if not remain:
                self.targets = []
                rospy.loginfo("[GraphExplorer] plan produced 0 targets")
                return
            cur = next(iter(remain))
            while remain:
                # cur에서 가장 가까운 점 고르기
                best = None; bd = 1e18
                for n in remain:
                    d = (n[0]-cur[0])**2 + (n[1]-cur[1])**2
                    if d < bd:
                        bd = d; best = n
                ordered.append(best)
                remain.remove(best)
                cur = best

        # 3) 예산 컷
        if self.budget_max_targets > 0 and len(ordered) > self.budget_max_targets:
            ordered = ordered[:self.budget_max_targets]

        self.targets = ordered
        rospy.loginfo(f"[GraphExplorer] plan='{self.explore_plan}', policy={self.order_policy}, "
                    f"gap={self.target_min_gap_m:.2f}m → targets={len(self.targets)}")


    # ---------- run ----------
    def run(self):
        try:
            rospy.wait_for_message(self.odom_topic, Odometry, timeout=10.0)
            # [수정] C++ 노드가 첫 메시지를 발행할 때까지 최대 15초 대기
            rospy.loginfo("[GraphExplorer] Waiting for the first /color_scan message...")
            rospy.wait_for_message(self.lidar_topic, PointCloud2, timeout=15.0)
            rospy.loginfo("[GraphExplorer] LiDAR topic is active. Starting exploration.")
        except rospy.ROSException as e:
            # [수정] 어떤 토픽에서 타임아웃이 발생했는지 명확히 표시
            rospy.logerr(f"[GraphExplorer] Failed to receive required message: {e}")
            return
        
        rospy.sleep(2.0)
        self.pub_resume.publish(Empty())
        self.pub_resume_goal.publish(Empty())
        self.explore()
        self.merge_and_save()
        rospy.signal_shutdown("done")


if __name__=="__main__":
    try:
        GraphExplorer().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in GraphExplorer: {e}")

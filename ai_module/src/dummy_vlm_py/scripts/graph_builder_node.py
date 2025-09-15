#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import os, json
from collections import defaultdict

import cv2
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from scipy.ndimage import binary_dilation, maximum_filter
from sensor_msgs.msg import PointCloud2


class GraphBuilder:
    """
    - traversable PointCloud2 -> free grid (inflate + morphology)
    - 8-connected graph (corner-cut 금지)
    - Target sets:
        * edge: 벽에서 [edge_clear_min, edge_clear_max] 밴드
        * skeleton: 거리변환(local maxima; 개활지 중심선)
        * interior: 개활지 균일 샘플(희박)
    - /tmp/free_mask.npz 저장
    - graph_data.json({ready:true, target_sets:{...}}) 원자적 쓰기
    """

    def __init__(self):
        rospy.init_node("graph_builder", anonymous=True)

        # Grid
        self.grid_resolution = rospy.get_param("~grid_resolution", 0.1)
        self.grid_width = rospy.get_param("~grid_width", 4000)
        self.grid_height = rospy.get_param("~grid_height", 4000)
        self.inflation_radius = rospy.get_param("~inflation_radius", 0.38)
        self.grid_origin_x = rospy.get_param("~grid_origin_x", -20.0)
        self.grid_origin_y = rospy.get_param("~grid_origin_y", -20.0)

        # Edge band
        self.edge_clear_min = rospy.get_param("~edge_clear_min", 0.05)
        self.edge_clear_max = rospy.get_param("~edge_clear_max", 1.20)
        self.target_stride_m = rospy.get_param("~target_stride_m", 0.40)

        # Skeleton (개활지 중심선)
        self.skel_clear_min = rospy.get_param("~skeleton_clear_min", 0.60)   # 중심선이 의미 있으려면 여유 필요
        self.skel_stride_m  = rospy.get_param("~skeleton_stride_m", 0.50)

        # Interior (희박 보강)
        self.int_clear_min = rospy.get_param("~interior_clear_min", 0.40)
        self.int_stride_m  = rospy.get_param("~interior_stride_m", 0.90)
        self.min_points_threshold = rospy.get_param("~min_points_threshold", 1)
        self.relax_if_sparse      = rospy.get_param("~relax_if_sparse", True)
        # file path (explorer와 공유)
        self.graph_file = rospy.get_param("~graph_file", "/tmp/graph_data.json")
        try:
            if os.path.exists(self.graph_file):
                os.remove(self.graph_file)
                rospy.loginfo(f"[GraphBuilder] removed stale {self.graph_file}")
        except Exception as e:
            rospy.logwarn(f"[GraphBuilder] cannot remove old json: {e}")

        # data
        self.inflated_grid = np.zeros((self.grid_height, self.grid_width), np.uint8)
        self.graph = defaultdict(list)
        self.target_sets = {"edge": [], "skeleton": [], "interior": []}

        # ROS
        self.pub_grid = rospy.Publisher("/exploration/traversable_graph", OccupancyGrid, queue_size=1)
        self.targets_pub = rospy.Publisher("/exploration/targets_markers", MarkerArray, queue_size=1, latch=True)
        self.edges_pub   = rospy.Publisher("/exploration/graph_edges", Marker, queue_size=1, latch=True)

        rospy.Subscriber("/traversable_area", PointCloud2, self.trav_cb)
        rospy.Subscriber("/state_estimation", Odometry, lambda _: None)

        self.dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        self.initial_done = False

        rospy.loginfo(f"[GraphBuilder] res={self.grid_resolution} infl={self.inflation_radius} "
                      f"edge=[{self.edge_clear_min},{self.edge_clear_max}] stride={self.target_stride_m}")

    # --- utils ---
    def world_to_grid(self, x, y):
        return (int((x - self.grid_origin_x) / self.grid_resolution),
                int((y - self.grid_origin_y) / self.grid_resolution))
    def grid_to_world(self, gx, gy):
        return (self.grid_origin_x + gx * self.grid_resolution,
                self.grid_origin_y + gy * self.grid_resolution)
    def valid(self, x, y):
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    # --- main ---
    def trav_cb(self, msg: PointCloud2):
        if self.initial_done: return
        try:
            rospy.loginfo("[GraphBuilder] Building…")
            free = self._points_to_free(msg)
            dt_m = cv2.distanceTransform(free, cv2.DIST_L2, 3) * self.grid_resolution
            vals = dt_m[free==1]
            if vals.size: rospy.loginfo(f"[GraphBuilder] clearance(m): min={vals.min():.2f} med={np.median(vals):.2f} max={vals.max():.2f}")

            # save mask
            np.savez_compressed("/tmp/free_mask.npz", free=free.astype(np.uint8),
                                res=self.grid_resolution, ox=float(self.grid_origin_x), oy=float(self.grid_origin_y))

            # graph
            self._build_graph(free)

            # targets
            self.target_sets["edge"]      = self._build_edge_targets(free, dt_m)
            self.target_sets["skeleton"]  = self._build_skeleton_targets(free, dt_m)
            self.target_sets["interior"]  = self._build_interior_targets(free, dt_m)

            # save
            # self._save_json(ready=True)
            # self._publish_debug(free)
            # # ★ RViz용 마커 발행
            # self.publish_target_markers()
            # self.publish_graph_edges()
            # self.initial_done = True
            self._save_json(ready=True)
            self.initial_done = True          # ← 먼저 re-entry 차단
            self._publish_debug(free)
            # ★ RViz용 마커 발행
            self.publish_target_markers()
            self.publish_graph_edges()



            rospy.loginfo(f"[GraphBuilder] Done. graph={len(self.graph)} "
                          f"edge={len(self.target_sets['edge'])} skel={len(self.target_sets['skeleton'])} "
                          f"int={len(self.target_sets['interior'])}")
        except Exception as e:
            rospy.logerr(f"[GraphBuilder] Error: {e}")

    # --- building blocks ---
    def _points_to_free(self, msg: PointCloud2):
        counts = np.zeros((self.grid_height, self.grid_width), np.int32)
        ps = msg.point_step; data = msg.data
        x_i=y_i=z_i=-1
        for f in msg.fields:
            if f.name=="x": x_i=f.offset
            elif f.name=="y": y_i=f.offset
            elif f.name=="z": z_i=f.offset

        N = msg.width*msg.height
        for i in range(N):
            o=i*ps
            x = np.frombuffer(data[o+x_i:o+x_i+4], np.float32)[0]
            y = np.frombuffer(data[o+y_i:o+y_i+4], np.float32)[0]
            z = np.frombuffer(data[o+z_i:o+z_i+4], np.float32)[0]
            if abs(z) < 0.1:
                gx,gy = self.world_to_grid(x,y)
                if self.valid(gx,gy): counts[gy,gx]+=1

        thr = max(1, int(self.min_points_threshold))     # ★ 더 관대하게
        occ = (counts >= thr).astype(np.uint8)           # occ==1 이면 “free 후보”

        # free 후보를 메꾸고 매끈하게 (unknown을 전부 장애물 취급하지 않음)
        free = occ.copy()
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), iterations=1)

        # 로봇 반경만큼 “안쪽으로” 여유(=거리변환 기반 침식)
        dt = cv2.distanceTransform(free, cv2.DIST_L2, 3) * self.grid_resolution
        safe = (dt >= max(0.20, self.inflation_radius*0.8)).astype(np.uint8)  # ★ 너무 보수적이면 0.8배

        # 너무 빈약하면 자동 완화
        if self.relax_if_sparse:
            total = int(safe.sum())
            maxclr = float(dt.max()) if dt.size else 0.0
            if total < 5000 or maxclr < 0.35:
                rospy.logwarn(f"[GraphBuilder] sparse free (pix={total}, max_clear={maxclr:.2f}) → relaxing")
                # 임계 낮추고, 닫기 강하게, 여유 기준도 완화
                thr2 = 1
                free2 = (counts >= thr2).astype(np.uint8)
                free2 = cv2.morphologyEx(free2, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), iterations=2)
                free2 = cv2.morphologyEx(free2, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), iterations=1)
                dt2 = cv2.distanceTransform(free2, cv2.DIST_L2, 3) * self.grid_resolution
                safe = (dt2 >= max(0.16, self.inflation_radius*0.6)).astype(np.uint8)
                dt = dt2  # 다음 단계에서 쓰게 교체

        # 내부 표현 업데이트
        self.inflated_grid = np.where(safe==1, 0, 100).astype(np.uint8)
        return safe


    def _build_graph(self, free: np.ndarray):
        self.graph.clear()
        H,W = free.shape
        ys,xs = np.where(free==1)
        for x,y in zip(xs,ys):
            nbr=[]
            for dx,dy in self.dirs:
                nx,ny=x+dx,y+dy
                if not (0<=nx<W and 0<=ny<H): continue
                if free[ny,nx]==0: continue
                if abs(dx)==1 and abs(dy)==1:  # no corner-cut
                    if not (free[y,nx]==1 and free[ny,x]==1): continue
                nbr.append((nx,ny))
            if nbr: self.graph[(x,y)] = nbr

    def _grid_stride(self, meters):
        return max(1, int(round(meters/self.grid_resolution)))

    def _sample_band_stride(self, mask: np.ndarray, stride_cells: int):
        ys,xs = np.where(mask)
        if xs.size==0: return []
        out=[]
        for x,y in zip(xs,ys):
            if (x%stride_cells==0) and (y%stride_cells==0):
                out.append((int(x),int(y)))
        return out

    def _build_edge_targets(self, free, dt_m):
        stride = self._grid_stride(self.target_stride_m)
        band = (free==1) & (dt_m>=self.edge_clear_min) & (dt_m<=self.edge_clear_max)
        t = self._sample_band_stride(band, stride)
        if not t:  # auto fallback
            vals = dt_m[free==1]
            if vals.size:
                c10, c50, c90 = np.percentile(vals,[10,50,90]).tolist()
                mn = max(0.05, min(0.15, c10))
                mx = max(0.6,  min(1.2, 0.5*c90+0.5*c50))
                band = (free==1) & (dt_m>=mn) & (dt_m<=mx)
                t = self._sample_band_stride(band, stride) or self._sample_band_stride(band, max(1,stride-1))
        return list(dict.fromkeys(t))

    def _build_skeleton_targets(self, free, dt_m):
        """거리변환 3x3 local maxima + 여유(clear_min) + stride"""
        stride = self._grid_stride(self.skel_stride_m)
        local_max = (dt_m == maximum_filter(dt_m, size=3))
        mask = (free==1) & local_max & (dt_m>=self.skel_clear_min)
        t = self._sample_band_stride(mask, stride)
        if not t:
            # 장면이 협소하면 자동 낮춤 (상위 70퍼센타일의 절반 정도)
            vals = dt_m[free==1]
            if vals.size:
                cmid = np.percentile(vals, 70) * 0.5
                mask2 = (free==1) & (dt_m >= max(0.15, cmid))
                t = self._sample_band_stride(mask2, stride)
        return list(dict.fromkeys(t))


    def _build_interior_targets(self, free, dt_m):
        stride = self._grid_stride(self.int_stride_m)
        mask = (free==1) & (dt_m>=self.int_clear_min)
        t = self._sample_band_stride(mask, stride)
        # edge/skeleton과 많이 겹치므로 수를 제한
        MAX_T=1500
        if len(t)>MAX_T: t=t[::len(t)//MAX_T]
        return list(dict.fromkeys(t))

    # --- IO ---
    def _publish_debug(self, free):
        grid = np.zeros((self.grid_height, self.grid_width), np.uint8)
        grid[free==1]=25
        for x,y in self.target_sets["edge"]: grid[y,x]=80
        for x,y in self.target_sets["skeleton"]: grid[y,x]=100
        for x,y in self.target_sets["interior"]: grid[y,x]=60
        msg = OccupancyGrid()
        msg.header.frame_id="map"; msg.header.stamp=rospy.Time.now()
        msg.info.resolution=self.grid_resolution
        msg.info.width=self.grid_width; msg.info.height=self.grid_height
        msg.info.origin.position.x=self.grid_origin_x; msg.info.origin.position.y=self.grid_origin_y
        msg.info.origin.orientation.w=1.0
        msg.data = grid.flatten().tolist()
        self.pub_grid.publish(msg)

    def _save_json(self, ready: bool):
        try:
            gdict = {f"({k[0]}, {k[1]})":[f"({n[0]}, {n[1]})" for n in v] for k,v in self.graph.items()}
            def ser(lst): return [f"({x}, {y})" for (x,y) in lst]
            data = {
                "ready": bool(ready),
                "traversable_graph": gdict,
                "grid_resolution": self.grid_resolution,
                "grid_origin_x": self.grid_origin_x,
                "grid_origin_y": self.grid_origin_y,
                "edge_clear_min": self.edge_clear_min,
                "edge_clear_max": self.edge_clear_max,
                "target_stride_m": self.target_stride_m,
                "skeleton_clear_min": self.skel_clear_min,
                "skeleton_stride_m": self.skel_stride_m,
                "interior_clear_min": self.int_clear_min,
                "interior_stride_m": self.int_stride_m,
                "target_sets": {
                    "edge": ser(self.target_sets["edge"]),
                    "skeleton": ser(self.target_sets["skeleton"]),
                    "interior": ser(self.target_sets["interior"]),
                },
                # 호환성(구버전 explorer가 읽어도 동작)
                "target_nodes": ser(self.target_sets["edge"])
            }
            tmp=self.graph_file+".tmp"
            with open(tmp,"w") as f: json.dump(data,f,indent=2)
            os.replace(tmp, self.graph_file)
            rospy.loginfo(f"[GraphBuilder] wrote {self.graph_file}")
        except Exception as e:
            rospy.logerr(f"[GraphBuilder] save error: {e}")


    def publish_target_markers(self):
        """edge / skeleton / interior 타깃을 색으로 구분해 MarkerArray로 발행"""
        ma = MarkerArray()
        now = rospy.Time.now()

        def make_sphere_list(ns, rgba, nodes, scale=0.08, z=0.02):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = ns
            m.id = hash(ns) & 0x7fffffff
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            # 그리드 -> 월드 변환해서 점 추가
            for (gx, gy) in nodes:
                wx, wy = self.grid_to_world(gx, gy)
                p = Point(x=wx, y=wy, z=z)
                m.points.append(p)
            return m

        # self.target_sets 사전에 edge/skeleton/interior 들어있음
        ma.markers.append(make_sphere_list("edge",      (1.0, 0.6, 0.0, 1.0),  self.target_sets.get("edge",      [])))
        ma.markers.append(make_sphere_list("skeleton",  (0.0, 0.8, 1.0, 1.0),  self.target_sets.get("skeleton",  [])))
        ma.markers.append(make_sphere_list("interior",  (0.6, 0.6, 0.6, 0.8),  self.target_sets.get("interior",  [])))

        self.targets_pub.publish(ma)

    def publish_graph_edges(self):
        """그래프의 간선들을 LINE_LIST로 발행 (중복 제거)"""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "graph_edges"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.01  # 1cm 굵기
        m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 1.0, 0.0, 0.7)

        seen = set()
        #for (x, y), nbrs in self.traversable_graph.items():
        for (x, y), nbrs in self.graph.items():
            for (nx, ny) in nbrs:
                if (nx, ny, x, y) in seen:  # 역간선 중복 방지
                    continue
                seen.add((x, y, nx, ny))
                wx1, wy1 = self.grid_to_world(x, y)
                wx2, wy2 = self.grid_to_world(nx, ny)
                m.points.append(Point(x=wx1, y=wy1, z=0.01))
                m.points.append(Point(x=wx2, y=wy2, z=0.01))

        self.edges_pub.publish(m)


if __name__ == "__main__":
    try:
        GraphBuilder(); rospy.spin()
    except rospy.ROSInterruptException:
        pass

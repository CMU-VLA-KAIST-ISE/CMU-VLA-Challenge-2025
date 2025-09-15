#!/usr/bin/env python3
import rospy
import numpy as np

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32MultiArray, Bool

import hashlib, struct
from collections import defaultdict

from planning_node.a_star_algorithm import A_star


class ExplorationNode:
    def __init__(self):
        rospy.init_node('exploration_node')
        
        self.mst_edges = []
        self.nodes = []
        self.new_data = True
        self.prev_edge_hash = None

        self.graph = defaultdict(list)
        self.traversal_order = []
        
        self.position = None
        self.node_to_travel = []

        self.grid_map = None

        self.cur_node_idx = None
        self.next_node_idx = None

        self.temporary_stop = 2.0
        
        # exploration switch
        self.explore_stop = False
        
        rospy.Subscriber("/state_estimation", Odometry, self.pose_callback)
        rospy.Subscriber("/mst_edges_marker", Marker, self.mst_callback)
        rospy.Subscriber("/node_list", PointCloud2, self.node_callback)
        rospy.Subscriber("/edge_list", Int32MultiArray, self.list_callback)
        rospy.Subscriber("/grid_map_info", OccupancyGrid, self.grid_map_callback)

        self.pose_pub = rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=1)
        self.pub_done = rospy.Publisher("/exploration_done", Bool, queue_size=1, latch=True)

        rospy.Timer(rospy.Duration(0.2), self.timer_callback)
        rospy.loginfo("Exploration node initialized. Listening to /mst_edges_marker")
    
    def timer_callback(self, event):
        if not self.explore_stop:
            self.waypoint_planning()

    def node_callback(self, msg):
        if self.new_data:
            self.nodes = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    
    def node_idx(self, node):
        if len(self.nodes) < 1:
            return None
        
        node_tuple = (round(node.x, 3), round(node.y, 3), round(node.z, 3))

        for i, n in enumerate(self.nodes):
            n_tuple = (round(n[0], 3), round(n[1], 3), round(n[2], 3))
            if node_tuple == n_tuple:
                return i
        
        rospy.logwarn(f"[WARNING] Node {node_tuple} not found in self.nodes.")
        return None
    
    def pose_callback(self, msg):
        self.latest_pose = msg
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
    
    def hash_list(self, msg):
        packed = struct.pack(f'{len(msg.data)}i', *msg.data)
        return hashlib.md5(packed).hexdigest()
    
    def list_callback(self, msg):
        current_hash = self.hash_list(msg)
        if current_hash == self.prev_edge_hash:
            self.new_data = False
            return
        rospy.loginfo("New edge_list received. load and process agian")
        self.new_data = True
        self.graph = defaultdict(list)
        self.prev_edge_hash = current_hash
    
    def grid_map_callback(self, msg):
        """
        /grid_map_info 토픽에서 OccupancyGrid 메시지를 수신했을 때 호출되는 콜백 함수입니다.
        수신된 데이터를 파싱하여 클래스 변수에 저장합니다.
        """
        if self.grid_map is None:
            # 1. 그리드 해상도(grid_size)를 메시지로부터 가져옵니다.
            self.grid_size = round(msg.info.resolution, 2)

            # 2. 그리드 원점(origin) 좌표를 메시지로부터 가져와 튜플 형태로 저장합니다.
            self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)

            # 3. 1차원 배열인 그리드 데이터를 2차원으로 변환합니다.
            # 메시지로부터 높이(height)와 너비(width) 정보를 가져옵니다.
            height = msg.info.height
            width = msg.info.width
            
            # 1차원 리스트 형태의 msg.data를 NumPy 배열로 변환합니다.
            grid_data_1d = np.array(msg.data, dtype=np.int8)
            
            # NumPy 배열을 (height, width) 크기의 2차원 배열로 재구성합니다.
            grid_data_2d = grid_data_1d.reshape((height, width))

            # 4. OccupancyGrid 값(0: 비점유, 100: 점유)을 원래 데이터 구조(0: 비점유, 1: 점유)로 변환합니다.
            # 먼저 모든 셀을 벽(1)으로 초기화한 uint8 타입의 맵을 생성합니다.
            self.grid_map = np.ones((height, width), dtype=np.uint8)
            
            # OccupancyGrid에서 비점유(0)였던 위치만 0으로 설정합니다.
            self.grid_map[grid_data_2d == 0] = 0

            row_indices, col_indices = np.where(self.grid_map == 0)
            traversable_cells = list(zip(row_indices, col_indices))
            self.total_free_cells = len(traversable_cells)

            self.a_star = A_star(self.grid_size, self.origin, self.grid_map, self.pose_pub)
            self.grid_to_world = self.a_star.grid_to_world
            self.world_to_grid = self.a_star.world_to_grid

            rospy.loginfo(f"Initial map created. Grid size: {self.grid_size}m, Total walkable cells: {self.total_free_cells}")

    def mst_callback(self, msg):
        if msg.type != Marker.LINE_LIST:
            rospy.logwarn("Received non LINE_LIST marker")
            return
        
        if len(msg.points) % 2 != 0:
            rospy.logwarn("LINE_LIST marker does not contain an even number of points")
            return
        
        if len(self.graph) < 1 :
            for i in range(0, len(msg.points), 2):
                p1 = msg.points[i]
                p2 = msg.points[i+1]

                # MST에서 정점 중심으로 한 방향의 간선만 추가
                self.graph[self.node_idx(p1)].append(self.node_idx(p2))
                self.graph[self.node_idx(p2)].append(self.node_idx(p1))
        
        if len(self.node_to_travel) < 1:
            self.explore_with_mst()
    
    def explore_with_mst(self):
        if len(self.graph) < 1 :
            return
        
        max_degree = max(len(self.graph[n]) for n in self.graph)
        candidate_roots = [n for n in self.graph if len(self.graph[n]) == max_degree]

        def distance(n1, n2):
            return np.linalg.norm(np.array(n1[:2]) - np.array(n2[:2]))

        root = min(candidate_roots, key=lambda n: distance(self.nodes[n], self.position))

        def dfs_traversal(graph, start_node):
            visited = set()
            traversal_order = []

            def dfs(node):
                visited.add(node)
                traversal_order.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor)
                        traversal_order.append(node)    # backtracking

            dfs(start_node)
            return traversal_order
        
        self.traversal_order = dfs_traversal(self.graph, root)
        self.node_to_travel = self.traversal_order.copy()

        rospy.loginfo(f"Node sequence: {self.node_to_travel}")
        # 대충 순회 순서를 구했으므로 각 순회 순서 별로 way point를 하나 씩 출력하고, state estimation과 비교해서 update 해야 함
    
    def waypoint_planning(self):    
        if self.new_data: # 새로운 데이터가 들어왔다면, 위의 코드를 처리할 때까지 대기
            return
        
        if len(self.node_to_travel) < 1: # self.node_to_travel이 계산될 때까지 대기
            return
                   
        if len(self.node_to_travel) == len(self.traversal_order):
            self.initial_time = rospy.get_time()
            self.next_node_idx = self.node_to_travel[0]
            
        move_done = self.a_star.move(self.position, self.nodes[self.next_node_idx])
        if move_done:
            rospy.sleep(self.temporary_stop)
            rospy.loginfo(f"[explore_node2] Temporary stop for {self.temporary_stop} seconds")

            if len(self.node_to_travel) == len(self.traversal_order):
                rospy.loginfo("[explore_node2] Initial node arrived, send waypoint following MST")

            self.cur_node_idx = self.node_to_travel.pop(0)
            if len(self.node_to_travel) > 1: # 모든 MST 노드를 돌지 않은 경우
                self.next_node_idx = self.node_to_travel[0]
                coverage_ratio = round((1 - len(self.node_to_travel) / len(self.traversal_order)) * 100, 2)
                rospy.loginfo(f"[explore_node2] {self.cur_node_idx} node reached! move to {self.next_node_idx} node. ({coverage_ratio}%, Elapsed: {rospy.get_time() - self.initial_time:.2f}s)")
            else: # 모든 MST 노드를 돈 경우
                self.explore_stop = True
                coverage_ratio = round((1 - len(self.node_to_travel) / len(self.traversal_order)) * 100, 2)
                rospy.loginfo(f"[explore_node2] All nodes reached! arrived at initial root node. ({coverage_ratio}%, Elapsed: {rospy.get_time() - self.initial_time:.2f}s)")
                self.pub_done.publish(Bool(data=True))
                rospy.loginfo("[explore_node2] /exploration_done published (True)")

if __name__ == '__main__':
    node = ExplorationNode()
    rospy.spin()
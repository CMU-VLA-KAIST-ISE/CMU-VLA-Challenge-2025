#!/usr/bin/env python3
import rospy
import numpy as np
# alias 에러 방지
np.int   = int
np.float = float
import networkx as nx
import heapq
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from std_msgs.msg import Int32MultiArray
import hashlib, struct

from planning_node.a_star_algorithm import A_star

def prim_mst_edges(G, nodes):
    N = len(nodes)

    visited = [False] * N
    hq = []
    mst_edges = []

    # 가장 많이 연결되어 있는 노드 중 가장 앞에 정렬되어 있는 것을 정점으로 선택
    
    visited[0] = True
    for j in range(N):
        if G.has_edge(nodes[0], nodes[j]):
            w = G[nodes[0]][nodes[j]]['weight']
            heapq.heappush(hq, (w, 0, j))
    
    # 가장 작은 간선 가중치 (거리)를 가진 v를 뽑아서 visited 에 추가, v에서의 연결된 간선도 heapq에 추가해서 반복
    while hq:
        w, u, v = heapq.heappop(hq)
        if visited[v]:
            continue
        visited[v] = True
        mst_edges.append((nodes[u], nodes[v]))
        for j in range(N):
            if not visited[j] and G.has_edge(nodes[v], nodes[j]):
                w2 = G[nodes[v]][nodes[j]]['weight']
                heapq.heappush(hq, (w2, v, j))

    return mst_edges


class MSTVisualizer:
    def __init__(self):
        rospy.init_node('mst_marker_visualizer')
        self.triggered = False
        self.G = None
        self.largest = None
        self.mst_edges = None
        self.nodes = None
        self.grid_map = None

        self.prev_edge_hash = None

        rospy.Subscriber("/node_list", PointCloud2, self.node_callback)
        rospy.Subscriber("/edge_list", Int32MultiArray, self.list_callback)
        rospy.Subscriber("/grid_map_info", OccupancyGrid, self.grid_map_callback)
        self.marker_pub = rospy.Publisher('/mst_edges_marker', Marker, queue_size=1)

    def hash_list(self, msg):
        # 바이너리 buffer 기준 hash 계산
        packed = struct.pack(f'{len(msg.data)}i', *msg.data)
        return hashlib.md5(packed).hexdigest()
    
    def node_callback(self, msg):
        if not self.triggered :
            self.nodes = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

    def list_callback(self, msg):
        self.edges = list(msg.data)
        current_hash = self.hash_list(msg)
        if current_hash == self.prev_edge_hash:
            return
        rospy.loginfo("[tsp_node] New edge_list received. load and process agian")
        self.triggered = False
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

            self.a_star = A_star(self.grid_size, self.origin, self.grid_map)

            self.triggered = True
            self.load_and_process_data()

    def load_and_process_data(self):
        if not self.triggered :
            return
        
        # MST를 돌리기 위한 그래프 생성
        self.G = nx.Graph()
        for i in range(0, len(self.edges), 2):
            a = self.edges[i]
            b = self.edges[i+1]
            path = self.a_star.a_star_search(self.nodes[a], self.nodes[b])
            if len(path) >= 2:
                self.G.add_edge(a, b, weight=len(path)-1)

            components = list(nx.connected_components(self.G))
            self.largest = sorted(list(max(components, key=len)))
            self.mst_edges = prim_mst_edges(self.G, self.largest)         
    
    def publish_mst(self):
        if self.mst_edges is None:
            return

        # print(f"MST edge 개수: {len(self.mst_edges)}")

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "mst_edges"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.8
        marker.color.a = 1.0

        for u, v in self.mst_edges:
            start = self.nodes[u]
            end = self.nodes[v]
            
            try:
                marker.points.append(Point(x=start[0], y=start[1], z=start[2]))
                marker.points.append(Point(x=end[0], y=end[1], z=end[2]))
            except Exception as e:
                rospy.logwarn(f"[tsp_node] No route for edge {u} <-> {v}, error as {e}")
        
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    visualizer = MSTVisualizer()
    rate = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        visualizer.publish_mst()
        rate.sleep()

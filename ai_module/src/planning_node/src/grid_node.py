#!/opt/conda/envs/leo/bin/python
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry, OccupancyGrid # Import the OccupancyGrid message
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Int32MultiArray, Header
import numpy as np
import hashlib
import networkx as nx

import cv2
from collections import deque

class GridNodePublisher:
    def __init__(self):
        rospy.init_node('grid_node_publisher')
        # --- 기본 파라미터 ---
        self.grid_size = rospy.get_param("~grid_size", 2.0)  # coarse grid size (m)
        self.a_star_node_size = rospy.get_param("~a_star_node_size", 0.4)  # uniform a* (fallback)
        self.min_points_per_grid = rospy.get_param("~min_points", 30)

        self.origin = np.array([0.0, 0.0, 0.0])

        self.received_pose = False

        self.prev_pc_hash = None
        self.prev_grid_msg = None
        self.prev_a_star_msg = None
        self.edge_msg = None

        rospy.Subscriber("/state_estimation", Odometry, self.pose_callback)
        rospy.Subscriber("/traversable_area", PointCloud2, self.pc_callback)
        self.grid_pub = rospy.Publisher("/node_list", PointCloud2, queue_size=1)
        self.a_node_pub = rospy.Publisher("/a_node_list", PointCloud2, queue_size=1)
        self.edge_pub = rospy.Publisher("/edge_list", Int32MultiArray, queue_size=1)
# ===============================================================================================================
        self.grid_map_info_pub = rospy.Publisher("/grid_map_info", OccupancyGrid, queue_size=1, latch=True)
# ===============================================================================================================

        # --- 적응형 파라미터 ---
        self.adaptive = rospy.get_param("~adaptive", False)
        self.subdivide_factor = rospy.get_param("~subdivide_factor", 2)     # 복잡 셀 세분화 배수 (2 or 3)
        self.anisotropy_thresh = rospy.get_param("~anisotropy_thresh", 2.0)  # λ1/λ2 임계
        self.min_points_frac = rospy.get_param("~min_points_frac", 0.2)     # 서브셀 허용 최소비율(원래 min_points 기준)

        # a* 간격 (복잡/개방)
        self.a_star_size_open = rospy.get_param("~a_star_size_open", 0.6)
        self.a_star_size_clutter = rospy.get_param("~a_star_size_clutter", 0.3)

    # ----------------- 유틸 -----------------
    def _anisotropy_ratio(self, pts_xy: np.ndarray) -> float:
        """2x2 공분산 고유값비 λ1/λ2 (λ1>=λ2). 띠 모양일수록 값이 큼."""
        if pts_xy.shape[0] < 5:
            return 1.0
        C = np.cov(pts_xy.T)
        # 수치 불안정 가드
        if not np.all(np.isfinite(C)):
            return 1.0
        vals, _ = np.linalg.eigh(C)
        vals = np.sort(vals)
        small = max(vals[0], 1e-9)
        return float(vals[1] / small)

    def _in_box_mask(self, pts_xy, x0, y0, x1, y1):
        return (pts_xy[:, 0] >= x0) & (pts_xy[:, 0] < x1) & (pts_xy[:, 1] >= y0) & (pts_xy[:, 1] < y1)

    # ----------------- 콜백 -----------------
    def pose_callback(self, msg):
        if not self.received_pose:
            # fixed origin (필요시 state_estimation에서 받아도 됨)
            self.origin = np.array([0, 0, 0.75])
            self.received_pose = True
            rospy.loginfo(f"[grid_node] Using initial origin from /state_estimation: {self.origin}")

    def hash_pointcloud(self, msg):
        return hashlib.md5(msg.data).hexdigest()

    def pc_callback(self, msg):
        if not self.received_pose:
            rospy.logwarn("[grid_node] Origin not yet received from /state_estimation.")
            return

        current_hash = self.hash_pointcloud(msg)
        if current_hash == self.prev_pc_hash:
            if self.prev_grid_msg and self.prev_a_star_msg:
                self.grid_pub.publish(self.prev_grid_msg)
                self.a_node_pub.publish(self.prev_a_star_msg)
                if self.edge_msg is not None:
                    self.edge_pub.publish(self.edge_msg)
            return
        
        # PointCloud 로드
        points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        points = np.array(points, dtype=np.float32)
        if points.shape[0] == 0:
            rospy.logwarn("[grid_node] Received empty point cloud.")
            return

        # 원점 보정
        points -= self.origin

        # --- coarse grid 분할 ---
        G = self.grid_size
        grid_indices = np.floor(points[:, :2] / G).astype(int)

        unique_grids = {}
        for idx, grid_idx in enumerate(grid_indices):
            key = (int(grid_idx[0]), int(grid_idx[1]))
            if key not in unique_grids:
                unique_grids[key] = []
            unique_grids[key].append(points[idx])

        # ----------------- 적응형 노드 생성 -----------------
        node_points = []
        fine_keys = []          # 노드 인덱스 == fine_keys 인덱스
        edge_presence = {}      # fine_key -> True
        clutter_cells = {}      # coarse key -> bool (복잡/단순)

        if self.adaptive:
            F = int(max(1, self.subdivide_factor))

            for key, pts in unique_grids.items():
                pts_arr = np.asarray(pts, dtype=np.float32)
                if pts_arr.shape[0] < max(5, int(self.min_points_per_grid * 0.1)):
                    continue

                pts_xy = pts_arr[:, :2]
                ratio = self._anisotropy_ratio(pts_xy)
                is_clutter = (ratio >= self.anisotropy_thresh)
                clutter_cells[key] = is_clutter

                f = F if is_clutter else 1
                cell_size = G / float(f)

                base_x = key[0] * G
                base_y = key[1] * G

                for i in range(f):
                    for j in range(f):
                        x0 = base_x + i * cell_size
                        y0 = base_y + j * cell_size
                        x1 = x0 + cell_size
                        y1 = y0 + cell_size
                        m = self._in_box_mask(pts_xy, x0, y0, x1, y1)
                        sub_pts = pts_arr[m]
                        if sub_pts.shape[0] >= max(5, int(self.min_points_per_grid * self.min_points_frac / (f * f))):
                            mean_xyz = np.mean(sub_pts, axis=0)
                            radius = 0.5 if f == 1 else 0.25
                            d = np.linalg.norm(sub_pts - mean_xyz, axis=1)
                            close_pts = sub_pts[d < radius]
                            safe_mean = np.mean(close_pts, axis=0) if close_pts.shape[0] > 0 else mean_xyz
                            node_points.append(safe_mean + self.origin)

                            fine_key = (key[0] * F + i, key[1] * F + j)
                            fine_keys.append(fine_key)
                            edge_presence[fine_key] = True
        else:
            # --- 기존 방식 (uniform grid) ---
            for key, pts in unique_grids.items():
                if len(pts) >= self.min_points_per_grid:
                    pts_arr = np.asarray(pts, dtype=np.float32)
                    mean_xyz = np.mean(pts_arr, axis=0)
                    d = np.linalg.norm(pts_arr - mean_xyz, axis=1)
                    close_pts = pts_arr[d < 0.5]
                    safe_mean = np.mean(close_pts, axis=0) if close_pts.shape[0] > 0 else mean_xyz
                    # 노드 추가
                    node_points.append(safe_mean + self.origin)
                    # fine_key는 coarse 키 자체를 사용 (F=1 가정)
                    fine_key = key
                    fine_keys.append(fine_key)
                    edge_presence[fine_key] = True
            # uniform 모드에선 clutter_cells 사용 안 함
            clutter_cells = {}

        # ----------------- 에지 생성 (fine grid 4-이웃) -----------------
        edge_set = set()
        key_to_index = {k: idx for idx, k in enumerate(fine_keys)}
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for k in fine_keys:
            idx = key_to_index[k]
            x, y = k
            for dx, dy in neighbor_offsets:
                nk = (x + dx, y + dy)
                if nk in key_to_index:
                    nidx = key_to_index[nk]
                    if nidx == idx:
                        continue
                    edge = (min(idx, nidx), max(idx, nidx))
                    edge_set.add(edge)

        edge_list = sorted([list(e) for e in edge_set], key=lambda x: (x[0], x[1]))
        flat_list = [i for e in edge_list for i in e]

        # --- 결과 체크 ---
        # edge로 node 간의 연결성 보장
        if len(node_points) == 0:
            rospy.logwarn("[grid_node] No valid node points found.")
            return
        
        self.G = nx.Graph()
        for i in range(0, len(flat_list), 2):
            a = flat_list[i]
            b = flat_list[i+1]
            self.G.add_edge(a, b)

        components = list(nx.connected_components(self.G))
        # 컴포넌트가 2개 이상이면(그래프가 나뉘어 있으면) 하나로 연결
        if len(components) > 1:    
            # 첫 번째 컴포넌트의 대표 노드를 하나 선택
            base_node = next(iter(components[0]))
            # 두 번째 컴포넌트부터 마지막 컴포넌트까지 순회
            for i in range(1, len(components)):
                # 현재 컴포넌트의 대표 노드를 하나 선택
                node_to_connect = next(iter(components[i]))
                # 첫 번째 컴포넌트의 노드와 현재 컴포넌트의 노드를 엣지로 연결
                self.G.add_edge(base_node, node_to_connect)
                flat_list += [base_node, node_to_connect]

        # ----------------- A* 노드 생성 -----------------
        # TODO: 이 부분 삭제하려고 했으나 이후에 자주 쓰이는 거 같아 일단 보류
        if self.adaptive:
            # 포인트별로 상위 coarse 셀의 복잡도에 따라 bin step 변경
            a_star_nodes = {}
            for p in points:
                gx, gy = np.floor(p[:2] / G).astype(int)
                is_clutter = clutter_cells.get((int(gx), int(gy)), False)
                step = self.a_star_size_clutter if is_clutter else self.a_star_size_open
                key = tuple(np.floor(p[:2] / step).astype(int))
                a_star_nodes.setdefault(key, []).append(p)

            a_star_node_points = []
            for pts in a_star_nodes.values():
                pts_arr = np.asarray(pts, dtype=np.float32)
                if pts_arr.shape[0] >= max(3, int(self.min_points_per_grid * 0.1)):
                    mean_xyz = np.mean(pts_arr, axis=0)
                    a_star_node_points.append(mean_xyz + self.origin)
        else:
            # 기존 uniform binning
            a_star_node_indices = np.floor(points[:, :2] / self.a_star_node_size).astype(int)
            unique_a_star_nodes = {}
            for idx, grid_idx in enumerate(a_star_node_indices):
                key = tuple(grid_idx)
                if key not in unique_a_star_nodes:
                    unique_a_star_nodes[key] = []
                unique_a_star_nodes[key].append(points[idx])

            a_star_node_points = []
            for pts in unique_a_star_nodes.values():
                pts_arr = np.asarray(pts, dtype=np.float32)
                if pts_arr.shape[0] >= self.min_points_per_grid:
                    mean_xyz = np.mean(pts_arr, axis=0)
                    d = np.linalg.norm(pts_arr - mean_xyz, axis=1)
                    close = pts_arr[d < 0.3]
                    if close.shape[0] > 0:
                        a_star_node_points.append(mean_xyz + self.origin)

        # --- 결과 체크 ---
        if len(a_star_node_points) == 0:
            rospy.logwarn("[grid_node] No valid a_node points found.")
            return
        
        # ----------------- 메시지 생성/발행 -----------------
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = msg.header.frame_id

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        grid_msg = pc2.create_cloud(header, fields, node_points)
        a_star_msg = pc2.create_cloud(header, fields, a_star_node_points)

        edge_msg = Int32MultiArray()
        edge_msg.data = flat_list

        # publish + cache
        self.edge_pub.publish(edge_msg)
        self.grid_pub.publish(grid_msg)
        self.a_node_pub.publish(a_star_msg)

        self.prev_grid_msg = grid_msg
        self.prev_a_star_msg = a_star_msg
        self.prev_pc_hash = current_hash
        self.edge_msg = edge_msg

        # rospy.loginfo(f"[grid_node] Published {len(node_points)} grid node(s), {len(a_star_node_points)} a-star node(s).")
        rospy.loginfo(f"[grid_node] Published {len(node_points)} grid node(s), {len(flat_list)//2} edge(s), {len(a_star_node_points)} a-star node(s).")

# ===============================================================================================================
        self.grid_size = calculate_grid_resolution(points)

        # 그리드 맵 생성 (point가 하나라도 들어있는 셀은 통행 가능하다고 간주)
        self.grid_map, self.origin, self.cell_points_map = create_occupancy_grid(points, self.grid_size)

        # 그리드 맵 스무스화
        self.grid_map = postprocess_grid_morphological(self.grid_map)

        # 통행 가능 셀로 처리된 벽 제거
        self.grid_map = mark_empty_cells_as_walls(self.grid_map, self.cell_points_map)

        # 모든 셀 간 연결성 보장
        self.grid_map = ensure_grid_connectivity(self.grid_map)

        # 3차원 좌표로 변환
        row_indices, col_indices = np.where(self.grid_map == 0)
        traversable_cells = list(zip(row_indices, col_indices))
        
        self.total_free_cells = len(traversable_cells)
        if self.total_free_cells == 0:
            rospy.logwarn("[grid_node] No traversable cells found.")
            return
        
        self.grid_map_info_msg = self.create_occupancy_grid_msg(header)
        self.grid_map_info_pub.publish(self.grid_map_info_msg)

        return

    def create_occupancy_grid_msg(self, header):
        """
        Generates an OccupancyGrid message from the class's grid map data.
        """
        grid_msg = OccupancyGrid()
        grid_msg.header = header
        
        # Set the map metadata
        grid_msg.info.resolution = self.grid_size
        grid_msg.info.width = self.grid_map.shape[1]
        grid_msg.info.height = self.grid_map.shape[0]
        
        # Set the origin of the map
        origin_pose = Pose()
        origin_pose.position.x = self.origin[0]
        origin_pose.position.y = self.origin[1]
        origin_pose.position.z = 0
        origin_pose.orientation.w = 1.0 # No rotation
        grid_msg.info.origin = origin_pose
        
        # Convert the grid map data to the required format
        # Your grid: 0 = free, 1 = occupied
        # OccupancyGrid: 0 = free, 100 = occupied, -1 = unknown
        # Create a new array with the same shape, initialized to -1 (unknown)
        occupancy_data = np.full(self.grid_map.shape, -1, dtype=np.int8)
        
        # Where your grid_map is 0 (free), set occupancy_data to 0
        occupancy_data[self.grid_map == 0] = 0
        
        # Where your grid_map is 1 (occupied), set occupancy_data to 100
        occupancy_data[self.grid_map == 1] = 100
        
        # Flatten the 2D array into a 1D list in row-major order and assign to message data
        grid_msg.data = occupancy_data.flatten().tolist()
        
        return grid_msg

    def grid_to_world(self, grid_coords, random=False, ddd=True): # TODO: 각 점들의 중심점으로 하는 방안
        """
        그리드 맵 좌표를 실제 세계 좌표로 변환합니다.
        cell_points_map에 해당 좌표의 포인트가 있으면, 그중 하나를 랜덤으로 반환합니다.
        없으면, 그리드 셀의 중심 좌표를 반환합니다.

        Args:
            grid_coords (tuple): 변환할 그리드 맵 좌표 (grid_row, grid_col).
            origin (tuple): 그리드 맵의 원점 (min_x, min_y).
            resolution (float): 그리드 맵의 해상도.
            cell_points_map (dict, optional): 각 셀에 포함된 포인트 리스트. 
                                            {(y, x): [point1, ...]} 형태.

        Returns:
            tuple: 변환된 실제 세계 좌표 (x, y).
        """
        grid_row, grid_col = grid_coords
        
        if random: # TODO: 왜 이거로 찍으면 가지 않고 멈춰있는지 알 수가 없다
            import random

            # 1. cell_points_map에 해당 셀의 포인트 정보가 있는지 확인합니다.
            # grid_coords는 (row, col)이므로 그대로 키로 사용합니다.
            if grid_coords in self.cell_points_map and self.cell_points_map[grid_coords]:
                # 2. 포인트 리스트가 있으면, 그중 하나를 선택합니다.
                random_point = random.choice(self.cell_points_map[grid_coords])
                # 선택된 포인트의 x, y 좌표를 반환합니다.

                if ddd: return random_point
                return random_point[:2]
            
        # --- 안정적인 좌표 계산 로직 ---
        
        # 1. 주변 셀의 벽 유무를 확인합니다.
        map_height, map_width = self.grid_map.shape
        
        # 안전한 경계 확인을 포함하여 벽 탐색
        # grid_map에서 1이 벽(occupied)이라고 가정합니다.
        wall_up = (grid_row > 0 and self.grid_map[grid_row - 1, grid_col] == 1)
        wall_down = (grid_row < map_height - 1 and self.grid_map[grid_row + 1, grid_col] == 1)
        wall_left = (grid_col > 0 and self.grid_map[grid_row, grid_col - 1] == 1)
        wall_right = (grid_col < map_width - 1 and self.grid_map[grid_row, grid_col + 1] == 1)
        
        # 2. 벽 위치에 따라 오프셋을 계산합니다.
        # 기본 위치는 셀의 중심입니다. 오프셋은 중심으로부터의 변위입니다.
        half_grid = self.grid_size / 2.0
        x_offset, y_offset = 0.0, 0.0
        
        # 좌우 벽 처리: 양쪽에 모두 벽이 있으면 움직이지 않습니다.
        if wall_left and not wall_right:
            x_offset = half_grid  # 왼쪽 벽을 피해 오른쪽으로 이동
        elif wall_right and not wall_left:
            x_offset = -half_grid # 오른쪽 벽을 피해 왼쪽으로 이동
            
        # 상하 벽 처리: 양쪽에 모두 벽이 있으면 움직이지 않습니다.
        if wall_up and not wall_down:
            y_offset = half_grid  # 위쪽 벽을 피해 아래쪽으로 이동
        elif wall_down and not wall_up:
            y_offset = -half_grid # 아래쪽 벽을 피해 위쪽으로 이동

        # 3. 최종 월드 좌표를 계산합니다.
        origin_x, origin_y = self.origin
        
        # 셀의 중심 좌표 계산
        center_x = (grid_col * self.grid_size) + origin_x + half_grid
        center_y = (grid_row * self.grid_size) + origin_y + half_grid
        
        # 중심 좌표에 계산된 오프셋을 더해 최종 좌표 결정
        world_x = center_x + x_offset
        world_y = center_y + y_offset
        
        if ddd: return np.array((world_x, world_y, 0))
        return np.array((world_x, world_y))
    
    def world_to_grid(self, world_coords):
        """
        실제 세계 좌표를 그리드 맵 좌표로 변환합니다.
        변환된 좌표가 맵을 벗어나거나 벽일 경우, 가장 가까운 통행 가능한 좌표를 찾아 반환합니다.

        Args:
            world_coords (tuple): 변환할 실제 세계 좌표 (x, y).
            origin (tuple): 그리드 맵의 원점 (min_x, min_y).
            resolution (float): 그리드 맵의 해상도.
            occupancy_grid (np.ndarray): 벽(1)과 통행 가능 공간(0)으로 이루어진 점유 격자 지도.

        Returns:
            tuple or None: 변환된 그리드 맵 좌표 (row, col). 통행 가능한 지점이 없으면 None을 반환합니다.
        """
        # 1. 초기 좌표 변환
        world_x, world_y, world_z = world_coords
        origin_x, origin_y = self.origin
        height, width = self.grid_map.shape
        
        grid_col = int((world_x - origin_x) / self.grid_size)
        grid_row = int((world_y - origin_y) / self.grid_size)

        # 2. 맵 경계 안으로 좌표 조정 (Clamping)
        # np.clip(값, 최소, 최대)는 값이 범위를 벗어나면 최소/최대값으로 고정합니다.
        clamped_row = np.clip(grid_row, 0, height - 1)
        clamped_col = np.clip(grid_col, 0, width - 1)
        
        # 3. 조정된 좌표가 통행 가능한지 확인
        if self.grid_map[clamped_row, clamped_col] == 0:
            return (clamped_row, clamped_col) # 통행 가능하면 바로 반환
            
        # 4. 통행 불가능하면(벽이면), 가장 가까운 통행 가능 지점을 BFS로 탐색
        queue = deque([(clamped_row, clamped_col)])
        visited = {(clamped_row, clamped_col)}
        
        while queue:
            current_row, current_col = queue.popleft()
            
            # 탐색 방향 (상, 하, 좌, 우)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_row, next_col = current_row + dr, current_col + dc
                
                # 맵 경계 체크 및 방문 여부 체크
                if 0 <= next_row < height and 0 <= next_col < width and (next_row, next_col) not in visited:
                    # 통행 가능한 지점을 찾았으면 즉시 반환
                    if self.grid_map[next_row, next_col] == 0:
                        return (next_row, next_col)
                    
                    # 벽이라면, 계속 탐색하기 위해 큐에 추가
                    visited.add((next_row, next_col))
                    queue.append((next_row, next_col))
                    
        # 5. 맵 전체를 탐색해도 통행 가능한 곳이 없는 경우
        return None

    
# ----------------- 유틸 -----------------
def get_passable_flat_edge_list(grid_map):
    """
    2D NumPy occupancy grid에서 '0'으로 표시된 통과 가능한 셀들을 노드로 간주하여
    인접한 노드 사이의 엣지 목록을 생성합니다.

    Args:
        grid_map (np.array): 2차원 NumPy 배열 (1은 벽, 0은 통과 가능).

    Returns:
        list: 엣지를 나타내는 인덱스들이 평탄화된 리스트.
    """
    # '0'인 셀들의 (y, x) 좌표를 fine_keys 리스트에 저장
    fine_keys = np.argwhere(grid_map == 0).tolist()
    
    # 딕셔너리를 사용하여 각 좌표를 고유한 인덱스에 매핑
    key_to_index = {tuple(k): idx for idx, k in enumerate(fine_keys)}
    
    # 4방향 이웃 (상, 하, 좌, 우)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    edge_set = set()
    
    # fine_keys의 각 셀에 대해 이웃 탐색
    for k in fine_keys:
        idx = key_to_index[tuple(k)]
        y, x = k  # NumPy의 argwhere는 (row, col), 즉 (y, x) 순서를 반환
        
        for dy, dx in neighbor_offsets:
            ny, nx = y + dy, x + dx
            nk = (ny, nx)
            
            # 이웃이 '0'인 셀이고, 아직 처리되지 않았다면
            if nk in key_to_index:
                nidx = key_to_index[nk]
                
                # 중복 엣지 방지를 위해 인덱스를 정렬하여 튜플로 저장
                edge = tuple(sorted((idx, nidx)))
                edge_set.add(edge)
    
    # 엣지 튜플을 리스트로 변환하고 정렬
    edge_list = sorted([list(e) for e in edge_set], key=lambda x: (x[0], x[1]))
    
    # 엣지 리스트를 평탄화 (flatten)
    flat_list = [i for e in edge_list for i in e]
    
    return flat_list

def create_occupancy_grid(points, grid_resolution):
    """
    통행 가능한 3D 포인트들로부터 점유 격자 지도와 각 셀의 포인트 목록을 생성합니다.
    
    결과:
    - grid: 벽(장애물) = 1, 통행 가능한 빈 공간 = 0
    - origin: 그리드의 월드 좌표계 기준 원점 (min_x, min_y)
    - cell_points_map: 각 셀에 포함된 포인트들을 저장한 딕셔너리 {(y, x): [point1, point2, ...]}
    """
    if points.shape[0] == 0:
        return np.ones((10, 10), dtype=np.uint8), (0, 0), {}
        
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
    
    width = int(np.ceil((max_x - min_x) / grid_resolution)) or 1
    height = int(np.ceil((max_y - min_y) / grid_resolution)) or 1
    
    grid = np.ones((height, width), dtype=np.uint8) 
    cell_points_map = {}
    
    for point in points:
        x, y, *_ = point
        grid_x = int((x - min_x) / grid_resolution)
        grid_y = int((y - min_y) / grid_resolution)
        
        grid_x = min(grid_x, width - 1)
        grid_y = min(grid_y, height - 1)

        if 0 <= grid_y and 0 <= grid_x:
            grid[grid_y, grid_x] = 0
            
            cell_key = (grid_y, grid_x) 

            if cell_key not in cell_points_map:
                cell_points_map[cell_key] = []
            cell_points_map[cell_key].append(point)
            
    return grid, (min_x, min_y), cell_points_map

def postprocess_grid_morphological(grid, kernel_size=3, iterations=1):
    """
    형태학적 '닫힘(Closing)' 연산을 이용해 그리드를 후처리합니다.
    지도 상의 끊어진 길을 연결하고 작은 장애물(노이즈)을 제거합니다.

    Args:
        grid (np.array): 점유 격자 지도 (0: 통행 가능, 1: 벽).
        kernel_size (int): 닫힘 연산에 사용할 커널(kernel)의 크기입니다.
                            이 값이 클수록 더 넓은 간격을 메울 수 있습니다.
        iterations (int): 닫힘 연산을 반복할 횟수입니다. 
                          더 강한 효과를 원할 때 값을 높입니다.

    Returns:
        np.array: 후처리된 점유 격자 지도.
    """
    # OpenCV 함수는 보통 흰색 객체(값이 255)를 대상으로 연산합니다.
    # 따라서 통행 가능(0)을 255로, 벽(1)을 0으로 임시 변환합니다.
    # (1 - grid)는 0->1, 1->0으로 바꾸고, 여기에 255를 곱해줍니다.
    # 데이터 타입은 uint8이어야 합니다.
    grid_for_cv = ((1 - grid) * 255).astype(np.uint8)

    # 형태학적 연산의 강도를 결정하는 커널(kernel)을 생성합니다.
    # kernel_size x kernel_size 크기의 사각형 커널입니다.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # '닫힘(Closing)' 연산을 적용합니다.
    # cv2.MORPH_CLOSE 플래그를 사용하여 닫힘 연산을 수행합니다.
    closed_grid = cv2.morphologyEx(grid_for_cv, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # 다시 원래의 그리드 형식으로 변환합니다. (통행 가능: 0, 벽: 1)
    # 255였던 부분을 0으로, 0이었던 부분을 1로 되돌립니다.
    processed_grid = 1 - (closed_grid / 255).astype(np.uint8)
    
    return processed_grid

def mark_empty_cells_as_walls(grid, cell_points_map):
    """
    cell_points_map에 포인트가 없는 셀을 벽(1)으로 표시합니다.

    Args:
    - grid (np.array): 원본 점유 격자 지도.
    - cell_points_map (dict): 각 셀의 좌표와 해당 셀에 포함된 포인트 목록.

    Returns:
    - modified_grid (np.array): 비어있는 셀이 벽으로 채워진 수정된 점유 격자 지도.
    """
    # 원본 수정을 방지하기 위해 그리드를 복사합니다.
    modified_grid = grid.copy()
    
    # 그리드의 높이(height)와 너비(width)를 가져옵니다.
    height, width = modified_grid.shape
    
    # 모든 그리드 셀을 하나씩 순회합니다.
    for y in range(height):
        for x in range(width):
            cell_key = (y, x)
            # 만약 cell_points_map에 현재 셀의 키가 존재하지 않는다면,
            # 이는 해당 셀에 포인트가 하나도 없다는 의미입니다.
            if cell_key not in cell_points_map:
                # 해당 셀을 벽(1)으로 설정합니다.
                modified_grid[y, x] = 1
                    
    return modified_grid

def ensure_grid_connectivity(grid):
    """
    점유 격자 지도의 모든 이동 가능 영역이 하나로 연결되도록 보장합니다.

    Args:
        grid (np.array): 점유 격자 지도 (0: 통행 가능, 1: 벽).

    Returns:
        np.array: 모든 이동 가능 영역이 연결된 새로운 지도.
    """
    # 1단계: 분리된 '섬'들 찾기
    components = _find_connected_components(grid)

    # 컴포넌트가 1개 이하면 이미 모두 연결된 상태
    if len(components) <= 1:
        return grid

    # 2단계: 가장 큰 '대륙' 결정하기 (크기 순으로 정렬)
    components.sort(key=len, reverse=True)
    
    mainland = components[0]
    islands = components[1:]
    
    connected_grid = grid.copy()

    # 3단계: '섬'들을 '대륙'에 연결하기
    for island in islands:
        # 가장 가까운 점 찾기
        p_island, p_mainland = _find_closest_points(island, mainland)
        
        # 두 점을 잇는 경로 생성 (다리 놓기)
        path = _draw_line(p_island, p_mainland)
        
        # 경로 상의 모든 셀을 이동 가능(0)으로 변경
        for y, x in path:
            connected_grid[y, x] = 0
            
        # 연결된 섬을 대륙에 편입시켜 다음 섬이 연결될 수 있도록 함
        mainland.extend(island)
        
    return connected_grid

def _draw_line(p1, p2):
    """두 점 (y1, x1)과 (y2, x2)를 잇는 그리드 셀들의 리스트를 반환합니다."""
    y1, x1 = p1
    y2, x2 = p2
    points = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((y1, x1))
        if y1 == y2 and x1 == x2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

# 헬퍼 함수: 두 컴포넌트 사이의 가장 가까운 두 점을 찾기
def _find_closest_points(comp1, comp2):
    """두 컴포넌트(셀 좌표 리스트) 사이의 가장 가까운 점 쌍을 찾습니다."""
    min_dist_sq = float('inf')
    closest_pair = (None, None)

    # 더 작은 컴포넌트를 바깥 루프로 두어 효율성 증대
    if len(comp1) > len(comp2):
        comp1, comp2 = comp2, comp1

    for p1 in comp1:
        for p2 in comp2:
            dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_pair = (p1, p2)
    return closest_pair

# 헬퍼 함수: 그리드에서 연결된 컴포넌트(섬)들을 찾기
def _find_connected_components(grid):
    """BFS를 사용하여 그리드 내의 모든 연결 요소를 찾습니다."""
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    components = []

    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0 and not visited[y, x]:
                # 새로운 컴포넌트 발견
                new_comp = []
                q = deque([(y, x)])
                visited[y, x] = True
                
                while q:
                    curr_y, curr_x = q.popleft()
                    new_comp.append((curr_y, curr_x))
                    
                    # 4방향 이웃 탐색
                    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ny, nx = curr_y + dy, curr_x + dx
                        
                        if (0 <= ny < height and 0 <= nx < width and
                                grid[ny, nx] == 0 and not visited[ny, nx]):
                            visited[ny, nx] = True
                            q.append((ny, nx))
                
                components.append(new_comp)
    
    return components

def calculate_grid_resolution(points, target_cell_count=10000):
    """
    포인트 클라우드 데이터(points)를 기반으로,
    총 그리드 셀의 수가 목표치(target_cell_count)에 가장 가까워지도록 하는
    최적의 grid_resolution을 계산합니다.
    
    계산된 grid_resolution은 항상 0.05의 배수입니다.

    Args:
        points (np.array): (N, 2) 또는 (N, 3) 형태의 포인트 클라우드 데이터.
        target_cell_count (int): 목표로 하는 총 그리드 셀의 수.

    Returns:
        float: 계산된 최적의 grid_resolution.
    """
    # 0. 포인트 데이터가 너무 적거나 범위가 없는 경우에 대한 예외 처리
    if points.shape[0] < 2:
        return 0.05 # 기본 해상도 반환

    # 1. 포인트 클라우드의 전체 범위(span) 계산
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_x == 0 or span_y == 0:
        return 0.05 # 면적이 없는 경우 기본 해상도 반환
        
    # 2. 목표 셀 개수를 만족하는 이상적인 해상도(ideal_resolution) 추정
    # (width * height) ≈ target_cell_count
    # (span_x / res) * (span_y / res) ≈ target_cell_count
    # res^2 ≈ (span_x * span_y) / target_cell_count
    area = span_x * span_y
    ideal_resolution = np.sqrt(area / target_cell_count)
    
    # 3. 이상적인 해상도와 가장 가까운 0.05 단위의 후보 해상도 2개 선정
    base_unit = 0.05
    # 이상적인 값보다 작거나 같은 0.05의 배수 중 가장 큰 값
    res_low = max(base_unit, np.floor(ideal_resolution / base_unit) * base_unit)
    # 이상적인 값보다 크거나 같은 0.05의 배수 중 가장 작은 값
    res_high = res_low + base_unit
    
    candidates = [res_low, res_high]
    best_resolution = -1
    min_diff = float('inf')

    # 4. 두 후보 해상도를 사용하여 실제 셀 개수를 계산하고, 목표치와 가장 가까운 것을 선택
    for res in candidates:
        width = int(np.ceil(span_x / res))
        height = int(np.ceil(span_y / res))
        current_cell_count = width * height
        
        diff = abs(current_cell_count - target_cell_count)
        
        if diff < min_diff:
            min_diff = diff
            best_resolution = res

    result =  max(round(best_resolution, 2), 0.10)     
    return result

# ===============================================================================================================

if __name__ == '__main__':
    try:
        GridNodePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

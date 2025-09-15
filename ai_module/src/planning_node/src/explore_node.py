#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool
import math

from planning_node.a_star_algorithm import A_star

# NOTE: WALL = 1, FREE = 0

class CoverHeuristicNode:
    def __init__(self):
        rospy.init_node('uncovered_area_node')
        
        # --- Configuration ---
        # 초 기준
        self.temporary_stop = 2.0
        self.stop_time = 60 * 9
        
        # self.grid_size에 곱해지는 값들
        self.update_multiplier = 20 # self.grid_size * self.update_multiplier # 최소한 이정도 이상 떨어진 프론티어를 찾도록 권장

        # 셀 간 멘하탄 거리 기준
        self.max_sight_distance = 20.0 # 방이 넓을 수록 더 멀리 볼 수 있다고 가정 None이면 무한대
        
        # --- State Variables ---
        self.grid_map = None

        self.state = 'Wait' # Exploration status flag
        self.current_pos = None
        self.current_grid_pos = None # Current position in grid coordinates
        self.target_waypoint = None # Current target waypoint

        self.initial_time = None

        self.covered = set() # Set to store all covered points
        self.coverage_ratio = 0

        # --- ROS Subscriber ---
        rospy.Subscriber("/state_estimation", Odometry, self.pose_callback)
        rospy.Subscriber("/grid_map_info", OccupancyGrid, self.grid_map_callback)
        
        # --- ROS Publisher ---
        self.pose_pub = rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=1)
        self.pub_done = rospy.Publisher("/exploration_done", Bool, queue_size=1, latch=True)

        self.state_timer = rospy.Timer(rospy.Duration(0.2), self.state_callback) # TODO: 조금 더 늘려서 실험 0.5s
        
    def state_callback(self, event):
        if self.state is None or self.grid_map is None or self.current_grid_pos is None:
            return
        # rospy.loginfo(f"[explore_node] ({self.state}) Coverage: {len(self.covered)}/{self.total_free_cells} ({coverage_ratio:.2%}, Elapsed: {rospy.get_time() - self.initial_time:.2f}s)")
        
        if self.state == 'Done':
            self.pub_done.publish(Bool(data=True))
            rospy.loginfo(f"[explore_node] ({self.state}) Exploration complete! All areas covered ({self.coverage_ratio:.2%}, Elapsed: {rospy.get_time() - self.initial_time:.2f}s)")
            rospy.signal_shutdown("")
            return

        visible_now = covered_points(self.grid_map, *self.current_grid_pos, max_distance=self.max_sight_distance)
        self.covered.update(visible_now)
        self.coverage_ratio = len(self.covered) / self.total_free_cells

        if self.state in ['Wait', 'Arrive']:
            if self.state != 'Wait':
                rospy.loginfo(f"[explore_node] ({self.state}) Target waypoint {self.world_to_grid(self.target_waypoint)} reached ({self.coverage_ratio:.2%}, Elapsed: {rospy.get_time() - self.initial_time:.2f}s)")
                self.target_waypoint = None
                rospy.loginfo(f"[explore_node] Temporary stop for {self.temporary_stop} seconds")
                rospy.sleep(self.temporary_stop)
            
            self.select_target()
            if self.target_waypoint is None:
                rospy.logwarn(f"[explore_node] ({self.state}) (state_callback) Target waypoint {self.target_waypoint} is None")

            if self.state == 'Wait':
                self.state_transition()
                return
        
        self.state_transition()
        return
        
    def state_transition(self):
        if self.coverage_ratio > 0.9999 or self.state == 'Done' or rospy.get_time() - self.initial_time > self.stop_time: # 완료 로직
            self.state = 'Done'
            return

        if self.target_waypoint is None:
            rospy.logwarn(f"[explore_node] ({self.state}) (state_transition) Target waypoint {self.target_waypoint} is None")
            rospy.logwarn(f"[explore_node] (state_callback) (state_transition) log both appear -> select_target() is wrong")
            rospy.logwarn(f"[explore_node] (state_transition) log only appear -> select_target() require more than 0.2 seconds")
            self.target_waypoint = np.array([0, 0, 0])

        self.state = 'Move'
        move_done = self.a_star.move(self.current_pos, self.target_waypoint)
        if move_done:
            self.state = 'Arrive'
        
        return

    def pose_callback(self, msg):
        """
        Called whenever the robot's pose is updated.
        Acts as the central control for making exploration decisions.
        """
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        # rospy.loginfo(f"[explore_node] {self.current_pose}")

        if self.initial_time is None:
            self.initial_time = rospy.get_time()

        if self.state is None or self.grid_map is None:
            return

        self.current_grid_pos = self.world_to_grid(self.current_pos) # (row, col)
        return

    def select_target(self):
        """Main logic to calculate and publish the next exploration target."""
        # **3. Calculate frontiers based on the entire covered area**
        if len(self.covered) == 0:
            self.covered.add(self.current_grid_pos)

        if self.coverage_ratio > 0.9999:
            self.state = 'Done'
            return

        frontiers = find_frontier_points(self.grid_map, self.covered)

        if len(frontiers) == 0:
            rospy.logwarn("[explore_node] find_frontier_points() is wrong")
            is_not_wall = (self.grid_map == 0)
            all_non_wall_coords = np.argwhere(is_not_wall)
            for coord in all_non_wall_coords:
                if tuple(coord) not in self.covered:
                    frontiers.update(tuple(coord))

            if len(frontiers) == 0:
                self.state = 'Done'
                return 

        # Choose the farthest frontier as the next target
        # target_grid_pos = find_farthest_frontier(frontiers, *self.current_grid_pos)
        # target_grid_pos = find_closest_frontier(frontiers, *self.current_grid_pos, grid_distance=self.grid_size, min_real_distance=self.update_distance)
        target_grid_pos = find_best_frontier_adaptive(frontiers, *self.current_grid_pos, self.grid_map, self.covered, self.grid_size, self.total_free_cells)

        if target_grid_pos is None:
            rospy.logwarn("[explore_node] select_frontier() is wrong.")
            if len(frontiers) > 0:
                target_grid_pos = frontiers[0]
            else:
                self.state = 'Done'
                return 
        
        # rospy.loginfo(f"[explore_node] ({self.state}) New Target waypoint: {target_grid_pos} Current: {self.current_grid_pos}")
        
        self.target_waypoint = self.grid_to_world(target_grid_pos)

        if self.target_waypoint is None:
            rospy.logwarn(f"[explore_node] ({self.state}) (select target) Target waypoint {self.target_waypoint} is None -> grid_to_world() is wrong.")
            self.target_waypoint = np.array([0, 0, 0])
    
        return
    
    def publish_move_command(self, current_position, target_waypoint):
        """
        Publishes a command for the robot to move to the given target waypoint.
        """
        self.state = 'Move'

        dx = target_waypoint[0] - current_position[0]
        dy = target_waypoint[1] - current_position[1]
        yaw = np.arctan2(dy, dx)

        pose2d = Pose2D()
        pose2d.x = target_waypoint[0]
        pose2d.y = target_waypoint[1]
        pose2d.theta = yaw

        self.pose_pub.publish(pose2d)
        # rospy.loginfo(f"[explore_node] ({self.state}) Destination ({pose_msg.x}, {pose_msg.y}) with heading {np.rad2deg(pose_msg.theta):.2f}°")
    # ========================================================================================== #

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
            grid_map = np.ones((height, width), dtype=np.uint8)
            
            # OccupancyGrid에서 비점유(0)였던 위치만 0으로 설정합니다.
            grid_map[grid_data_2d == 0] = 0

            row_indices, col_indices = np.where(grid_map == 0)
            traversable_cells = list(zip(row_indices, col_indices))
            self.total_free_cells = len(traversable_cells)

            self.a_star = A_star(self.grid_size, self.origin, grid_map, self.pose_pub)
            self.grid_to_world = self.a_star.grid_to_world
            self.world_to_grid = self.a_star.world_to_grid

            rospy.loginfo(f"[explore_node] ({self.state}) Initial map created. Grid size: {self.grid_size}m, Total walkable cells: {self.total_free_cells}")

            self.grid_map = grid_map 
    # ========================================================================================== #

# ========================================================================================== #
def find_frontier_points(grid_map, visible_points):
    """
    보이는 지점(visible_points)과 보이지 않는 길의 경계, 즉 프론티어를 찾습니다.
    프론티어는 '보이는 길'에 인접한 '보이지 않는 길'입니다.
    """
    frontiers = set()
    height, width = len(grid_map), len(grid_map[0])

    # 보이는 각 지점에 대해 반복합니다.
    for r, c in visible_points:
        # 보이는 지점이 '길'이 아니면(벽이면) 건너뜁니다.
        # 일반적으로 visible_points는 길만 포함하지만, 안전을 위한 확인입니다.
        if grid_map[r][c] != 0:
            continue

        # 해당 지점의 상하좌우 인접 지점을 확인합니다.
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # 인접 지점이 '보이지 않는 길'인지 확인하는 조건
            if (0 <= nr < height and 0 <= nc < width and  # a. 맵 범위 안이고,
                grid_map[nr][nc] == 0 and                   # b. 벽이 아니면서(길이면서),
                (nr, nc) not in visible_points):            # c. 보이지 않는 곳이라면
                
                frontiers.add((nr, nc))

    return frontiers

def covered_points(grid_map, start_row, start_col, max_distance=None):
    """
    주어진 2D 그리드 맵에서 특정 지점 (row, col)의 시야(Field of View, FOV)를 계산합니다.
    max_distance가 주어지면 해당 거리까지만 시야를 제한합니다.

    Args:
        grid_map (list[list[int]]): 0(이동 가능)과 1(벽)으로 이루어진 2D 맵.
        start_row (int): 관찰자의 시작 위치 (행).
        start_col (int): 관찰자의 시작 위치 (열).
        max_distance (int, optional): 최대 시야 거리. 기본값은 None (무제한).
    """
    # 시작 지점이 맵 범위를 벗어나거나 벽 안인 경우 예외 처리
    if not (0 <= start_row < len(grid_map) and 0 <= start_col < len(grid_map[0])):
        return set()
    if grid_map[start_row][start_col] == 1:
        return set()

    # 보이는 지점을 저장할 집합(set), 시작점은 항상 보임
    visible = set([(start_row, start_col)])
    
    # 시작점을 중심으로 8개의 옥탄트(octant, 45도 영역)를 각각 스캔
    for octant in range(8):
        _scan_octant(
            origin_i=start_row, origin_j=start_col, octant=octant,
            grid_map=grid_map, visible_points=visible,
            max_distance=max_distance,
            row_depth=1, start_slope=1.0, end_slope=0.0
        )
    return visible

def _scan_octant(origin_i, origin_j, octant, grid_map, visible_points, max_distance, row_depth, start_slope, end_slope):
    """
    재귀적으로 하나의 옥탄트를 스캔하여 보이는 지점을 찾습니다.
    """
    # ★★★ 거리 제한 로직 수정 ★★★
    # max_distance가 None이 아니고, 현재 탐색 깊이가 최대 거리를 벗어나면 재귀를 종료합니다.
    if max_distance is not None and row_depth > max_distance:
        return

    height, width = len(grid_map), len(grid_map[0])

    if start_slope < end_slope:
        return

    prev_was_wall = None
    min_col = round(row_depth * end_slope)
    max_col = round(row_depth * start_slope)

    for col in range(max_col, min_col - 1, -1):
        di, dj = _transform_octant(row_depth, col, octant)
        current_i, current_j = origin_i + di, origin_j + dj

        if not (0 <= current_i < height and 0 <= current_j < width):
            continue

        is_wall = (grid_map[current_i][current_j] == 1)
        if not is_wall:
            visible_points.add((current_i, current_j))

        # --- 섀도캐스팅(그림자 드리우기) 핵심 로직 ---
        if is_wall:
            if prev_was_wall is False:
                new_end_slope = (col + 0.5) / (row_depth - 0.5)
                _scan_octant(
                    origin_i, origin_j, octant, grid_map, visible_points,
                    max_distance,
                    row_depth + 1, start_slope, new_end_slope
                )
            prev_was_wall = True
        else:
            if prev_was_wall is True:
                start_slope = (col + 0.5) / (row_depth + 0.5)
            prev_was_wall = False
    
    if prev_was_wall is False:
        _scan_octant(
            origin_i, origin_j, octant, grid_map, visible_points,
            max_distance,
            row_depth + 1, start_slope, end_slope
        )

def _transform_octant(row, col, octant):
    """로컬 옥탄트 좌표를 글로벌 맵 좌표 오프셋으로 변환합니다."""
    if octant == 0: return (-row, col)
    if octant == 1: return (-col, row)
    if octant == 2: return (col, row)
    if octant == 3: return (row, col)
    if octant == 4: return (row, -col)
    if octant == 5: return (col, -row)
    if octant == 6: return (-col, -row)
    if octant == 7: return (-row, -col)
    return (0, 0)

def find_best_frontier(frontiers, i, j, grid_map, covered, 
                      grid_distance=1.0, weights=None, total_map_size=10000):
    """
    여러 기준을 종합적으로 고려하여 최적의 프론티어를 선택합니다.
    
    Args:
        frontiers (list): 후보 지점들의 리스트 [(x1, y1), (x2, y2), ...].
        i (int): 현재 위치의 그리드 x 좌표.
        j (int): 현재 위치의 그리드 y 좌표.
        grid_map (np.ndarray): 2D occupancy grid (1=벽, 0=통행가능).
        covered (set): 방문한 지점들의 집합 {(row, col), ...}.
        grid_distance (float): 그리드 한 칸의 실제 거리.
        weights (dict): 각 기준의 가중치 {'info_gain': 0.4, 'distance': 0.3, 'wall_distance': 0.3}.
        
    Returns:
        tuple: 최적의 프론티어 (x, y) 또는 None.
    """
    if not frontiers:
        return None
    
    # 기본 가중치 설정
    if weights is None:
        weights = {
            'info_gain': 0.5,      # 정보 이득
            'distance': 0.5,       # 거리 (가까울수록 좋음)
            'wall_distance': 0,  # 벽과의 거리 (멀수록 좋음)
            'exploration': 0     # 미탐색 영역 근접도
        }
    
    # 벽과의 거리 맵 미리 계산 (벽=1에서 거리 계산)
    # wall_distance_map = distance_transform_edt(1 - grid_map)
    
    best_frontier = None
    best_score = float('-inf')
    
    for frontier in frontiers:
        fx, fy = frontier
        
        # 1. 정보 이득 계산
        info_gain_score = calculate_info_gain(fx, fy, covered, grid_map, total_map_size)
        
        # 2. 거리 점수 계산 (가까울수록 높은 점수)
        distance = np.linalg.norm(np.array(frontier[:2]) - np.array([i, j])) * grid_distance
        distance_score = calculate_distance_score(distance, max_preferred_distance=grid_distance*10)
        
        # 3. 벽과의 거리 점수 계산 (안전성)
        # wall_distance_score = calculate_wall_distance_score(fx, fy, wall_distance_map)
        wall_distance_score = 0
        
        # 4. 미탐색 영역 근접도 점수
        #  exploration_score = calculate_exploration_score(fx, fy, covered, grid_map)
        exploration_score = 0
        
        # 총 점수 계산
        total_score = (weights['info_gain'] * info_gain_score + 
                      weights['distance'] * distance_score + 
                      weights['wall_distance'] * wall_distance_score + 
                      weights['exploration'] * exploration_score)
        
        if total_score > best_score:
            best_score = total_score
            best_frontier = frontier
    
    return best_frontier

def calculate_info_gain(x, y, covered, grid_map, total_map_size):
    """
    해당 위치에서 얻을 수 있는 정보 이득을 계산합니다.
    covered_points는 전역함수로 가정합니다.
    """
    try:
        # 해당 위치에서 볼 수 있는 지점들 (전역함수 covered_points 사용)
        visible = covered_points(grid_map, x, y, 20.0)  # row, col 순서 주의
        
        # 새로 볼 수 있는 지점의 수 (아직 방문하지 않은 지점)
        new_visible = len([p for p in visible if p not in covered])
        
        # 정규화 (0~1 범위로)
        max_possible_visible = total_map_size
        return new_visible / max_possible_visible if max_possible_visible > 0 else 0
        
    except Exception:
        return 0.0

def calculate_distance_score(distance, max_preferred_distance=50.0):
    """
    거리에 따른 점수를 계산합니다. max_preferred_distance에 가까울수록 높은 점수.
    """
    if distance < 0:
        return 0.0

    # max_preferred_distance에서 멀어질수록 점수가 감소하도록 조정
    # 거리 차이의 절댓값을 사용하여 대칭적으로 점수 계산
    distance_diff = abs(distance - max_preferred_distance)
    
    # 지수 감소 함수 사용 (거리 차이가 0에 가까울수록 1에 가까워짐)
    return math.exp(-distance_diff / max_preferred_distance)

# def calculate_wall_distance_score(x, y, wall_distance_map):
#     """
#     벽과의 거리 기반 안전성 점수를 계산합니다.
#     """
#     if y < 0 or y >= wall_distance_map.shape[0] or x < 0 or x >= wall_distance_map.shape[1]:
#         return 0.0
    
#     wall_distance = wall_distance_map[y, x]
    
#     # 벽에서 3칸 이상 떨어져 있으면 최대 점수
#     min_safe_distance = 3.0
#     if wall_distance >= min_safe_distance:
#         return 1.0
    
#     # 선형적으로 점수 감소
#     return wall_distance / min_safe_distance

# def calculate_exploration_score(r, c, covered, grid_map, search_radius=10):
#     """
#     주변 미탐색 영역의 밀도를 기반으로 탐색 가치를 계산합니다.
#     """
#     unexplored_count = 0
#     total_count = 0
    
#     # 주변 영역을 검사
#     for dr in range(-search_radius, search_radius + 1):
#         for dc in range(-search_radius, search_radius + 1):
#             nr, nc = r + dr, c + dc
            
#             # 맵 경계 체크
#             if (0 <= nr < grid_map.shape[0] and 
#                 0 <= nc < grid_map.shape[1] and 
#                 grid_map[nr, nc] == 0):  # 통행 가능한 지역만
                
#                 total_count += 1
#                 if (nr, nc) not in covered:
#                     unexplored_count += 1
    
#     if total_count == 0:
#         return 0.0
    
#     return unexplored_count / total_count

# 사용 예시 및 유틸리티 함수들
def adaptive_weights(covered_ratio):
    """
    탐색 진행도에 따라 가중치를 동적으로 조정합니다.
    
    Args:
        covered_ratio (float): 전체 맵 대비 탐색된 비율 (0.0 ~ 1.0)
    """
    if covered_ratio < 0.5:  # 초기 탐색 단계
        return {
            'info_gain': 0.5,
            'distance': 0.5,
            'wall_distance': 0,
            'exploration': 0
        }
    elif covered_ratio < 0.75:  # 중간 탐색 단계
        return {
            'info_gain': 0.3,
            'distance': 0.7,
            'wall_distance': 0,
            'exploration': 0
        }
    else:  # 마무리 탐색 단계
        return {
            'info_gain': 0.1,
            'distance': 0.9,
            'wall_distance': 0,
            'exploration': 0
        }

def find_best_frontier_adaptive(frontiers, i, j, grid_map, covered, 
                               grid_distance=1.0, total_map_size=10000):
    """
    탐색 진행도에 따라 가중치를 자동으로 조정하는 버전
    """
    covered_ratio = len(covered) / total_map_size if total_map_size > 0 else 0
    adaptive_weights_dict = adaptive_weights(covered_ratio)
    
    return find_best_frontier(frontiers, i, j, grid_map, covered,
                             grid_distance, adaptive_weights_dict, total_map_size)

def find_closest_frontier(frontiers, i, j, grid_distance, min_real_distance):
    """
    후보 지점들(frontiers) 중에서 현재 위치(i, j)와의 실제 맵 상 거리를 기준으로
    가장 가까운 지점을 찾습니다.
    단, min_real_distance 이상 떨어진 지점을 우선적으로 고려합니다.

    Args:
        frontiers (list): 후보 지점들의 리스트 [(x1, y1), (x2, y2), ...].
        i (int): 현재 위치의 그리드 x 좌표.
        j (int): 현재 위치의 그리드 y 좌표.
        min_real_distance (float): 고려할 최소 실제 거리 (예: 20미터).
        grid_distance (float): 그리드 한 칸이 나타내는 실제 거리 (예: 5미터).

    Returns:
        tuple: 찾은 가장 가까운 지점 (x, y) 또는 후보가 없으면 None.
    """
    # 후보 지점이 없는 경우 None을 반환합니다.
    if not frontiers:
        return None

    # 최종적으로 반환할 후보 지점과 실제 거리(의 제곱)를 저장할 변수
    closest_point_overall = None
    min_dist_real_sq_overall = float('inf')

    # min_real_distance 조건을 만족하는 후보 지점과 실제 거리(의 제곱)를 저장할 변수
    closest_point_filtered = None
    min_dist_real_sq_filtered = float('inf')
    
    # 최소 실제 거리의 제곱 (계산 최적화를 위해 미리 계산)
    min_real_distance_sq_threshold = min_real_distance**2

    # 모든 후보 지점을 순회하며 조건에 맞는 가장 가까운 지점을 찾습니다.
    for point in frontiers:
        px, py = point
        
        # 1. 그리드 좌표상 거리의 제곱을 계산합니다.
        distance_grid_sq = (px - i)**2 + (py - j)**2
        
        # 2. 실제 맵 상의 거리의 제곱을 계산합니다.
        # (실제 거리)^2 = (그리드 거리 * grid_distance)^2
        #             = (그리드 거리)^2 * (grid_distance)^2
        distance_real_sq = distance_grid_sq * (grid_distance**2)
        
        # 3. 전체 후보 중 가장 가까운 지점을 항상 추적합니다. (Fallback 용도)
        if distance_real_sq < min_dist_real_sq_overall:
            min_dist_real_sq_overall = distance_real_sq
            closest_point_overall = point

        # 4. min_real_distance 이상 떨어진 지점들 중에서 가장 가까운 지점을 찾습니다.
        if distance_real_sq >= min_real_distance_sq_threshold and distance_real_sq < min_dist_real_sq_filtered:
            min_dist_real_sq_filtered = distance_real_sq
            closest_point_filtered = point
            
    # min_real_distance 조건을 만족하는 지점이 있으면 그 지점을 반환하고,
    # 없으면 전체 중에서 가장 가까웠던 지점을 반환합니다.
    if closest_point_filtered is not None:
        return closest_point_filtered
    else:
        return closest_point_overall

def find_farthest_frontier(frontiers, i, j):
    """
    후보 지점들(frontiers) 중에서 현재 위치(i, j)와 가장 가까운 지점을 찾습니다.
    """
    # 후보 지점이 없는 경우 None을 반환합니다.
    if not frontiers:
        return None

    closest_point = None
    min_distance_sq = 0  # 최소 거리를 저장하기 위한 변수, 무한대로 초기화

    # 모든 후보 지점을 순회하며 가장 가까운 지점을 찾습니다.
    for point in frontiers:
        px, py = point
        
        # 유클리드 거리의 제곱을 계산합니다.
        # 실제 거리를 비교하는 것이 아니므로 제곱근 계산(sqrt)은 생략하여 계산 속도를 높입니다.
        distance_sq = (px - i)**2 + (py - j)**2
        
        # 현재까지의 최소 거리보다 더 가까운 지점을 찾으면 업데이트합니다.
        if distance_sq > min_distance_sq:
            min_distance_sq = distance_sq
            closest_point = point
            
    return closest_point

if __name__ == '__main__':
    node = CoverHeuristicNode()
    rospy.spin()
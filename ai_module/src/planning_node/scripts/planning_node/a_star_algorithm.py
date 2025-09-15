#!/usr/bin/env python3
import rospy
import heapq
import numpy as np
from collections import deque
from geometry_msgs.msg import Pose2D

class A_star():
    def __init__(self, grid_size, origin, grid_map, pose_pub=None):
        self.grid_size = grid_size
        self.origin = origin
        self.original_grid_map = np.copy(grid_map) # 원본 맵을 저장
        self.grid_map = grid_map

        self.pose_pub = pose_pub
        self.current_route = None
        self.current_destination = None
        self.route_idx = 0

    def a_star_search(self, any_start, any_end, return_world=False, turn_penalty=0.01):
        """
        Finds the shortest path from start to end on a grid using the A* algorithm.
        It avoids obstacles (marked as 1) and does not consider diagonal movement.
        It prefers paths with fewer turns by adding a penalty for changing direction.

        Args:
            grid (np.array): A 2D array where 0 is walkable and 1 is an obstacle.
            start (tuple): The starting coordinates (row, col).
            end (tuple): The ending coordinates (row, col).
            turn_penalty (float): Additional cost for making a turn.

        Returns:
            list: A list of tuples representing the path from start to end.
                Returns None if no path is found.
        """
        if any_start is None:
            rospy.logwarn(f"start position {any_start} is None")
        if any_end is None:
            rospy.logwarn(f"start position {any_end} is None")
            any_end = np.array([0, 0, 0])

        if len(any_start) == 3: # world 좌표를 넣었을 때 안전한 벽이 아닌 가까운 셀로 변환
            start = self.world_to_grid(any_start)
        else:
            start = any_start # grid 좌표로 
        if len(any_end) == 3:
            end = self.world_to_grid(any_end)
        else:
            end = any_end # grid 좌표로

        if np.array_equal(start, end):
            if return_world:
                temp = self.grid_to_world(start)
                return [temp, temp]
            return [start, end]

        if self.grid_map[start[0]][start[1]] == 1:
            rospy.logwarn(f"a_star_search() get start cell {start} as wall. world_to_gird() is wrong")
            return []
        if self.grid_map[end[0]][end[1]] == 1:
            rospy.logwarn(f"a_star_search() get end cell {end} as wall. world_to_gird() is wrong")
            return []

        def reconstruct_path(came_from, current):
            total_path = [current]
            while current in came_from:
                current = came_from[current]
                total_path.append(current)

            final_path = total_path[::-1]
            if not np.array_equal(final_path[0], start) or not np.array_equal(final_path[-1], end):
                rospy.logwarn(f"a_star_search() must return path including start and destination")
                if return_world:
                    return [self.grid_to_world(start), self.grid_to_world(end)]
                return [start, end]
            
            if return_world:
                return [self.grid_to_world(p) for p in final_path]

            return final_path

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        
        g_score = { (r, c): float('inf') for r in range(self.grid_map.shape[0]) for c in range(self.grid_map.shape[1]) }
        g_score[start] = 0
        
        f_score = { (r, c): float('inf') for r in range(self.grid_map.shape[0]) for c in range(self.grid_map.shape[1]) }
        f_score[start] = heuristic(start, end)
        
        open_set_hash = {start}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == end:
                return reconstruct_path(came_from, current)

            for move in neighbors:
                neighbor = (current[0] + move[0], current[1] + move[1])
                
                if not (0 <= neighbor[0] < self.grid_map.shape[0] and 0 <= neighbor[1] < self.grid_map.shape[1]):
                    continue
                if self.grid_map[neighbor[0]][neighbor[1]] == 1:
                    continue
                
                # --- 턴 페널티 로직 시작 ---
                cost = 1  # 기본 이동 비용
                
                # came_from에 current가 있어야 이전 노드를 알 수 있음 (시작 노드는 제외)
                if current in came_from:
                    parent = came_from[current]
                    # 이전 방향 벡터: (parent -> current)
                    prev_move = (current[0] - parent[0], current[1] - parent[1])
                    
                    # 현재 방향 벡터와 이전 방향 벡터가 다르면(턴) 페널티 추가
                    if move != prev_move:
                        cost += turn_penalty
                
                tentative_g_score = g_score[current] + cost
                # --- 턴 페널티 로직 종료 ---
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        rospy.logwarn(f"a_star_search() has error")             
        return []
    
    def distance_okay(self, destination, multiplier=1, arrive=False):
        threshold = self.distance_threshold * multiplier
        if arrive:
            threshold = self.arrive_threshold * multiplier

        distance = np.linalg.norm(self.current_pos[:2] - destination[:2])
        if distance < threshold:
            return True
        return False
    
    def move(self, current_pos, real_destination, arrive_add=0.20, move_mul=4.5):
        """
        개선된 이동 함수
        - 경로 캐싱으로 안정성 향상
        - 명확한 웨이포인트 선택 로직
        - 상태 관리 개선
        """
        current_time = rospy.get_time()
        if self.current_route is None:
            self.last_route_update = rospy.get_time()
        self.current_pos = current_pos
        self.arrive_threshold = max(self.grid_size + arrive_add, 0.3) # 로봇의 길이 / 2
        # self.distance_threshold = self.grid_size * move_mul
        self.distance_threshold = 0.49 # TODO: 조금 더 늘려서 실험 1.0m

        # 1. 최종 목적지 도착 확인 (최우선)
        destination = self.grid_to_world(self.world_to_grid(real_destination))
        if self.distance_okay(destination, multiplier=1, arrive=True):
            self._reset_route()
            return True
        
        distance = np.linalg.norm(current_pos[:2] - destination[:2])
        if current_time - self.last_route_update > distance * 12: # 거리에 따른 예상 시간보다 초과했을 때
            if current_time - self.last_route_update > distance * 20 or self.distance_okay(destination, multiplier=1.15, arrive=True): # 적당히 가깝거나 너무 오래 걸리면 도착으로 침
                self._reset_route()
                return True
            else:
                self._update_route(destination) # 멀다면 경로 재계획
                self.last_route_update = current_time
        
        # 2. 경로 업데이트 필요성 확인
        if self._should_update_route(current_time):
            self._update_route(destination)
            self.last_route_update = current_time
        
        # 4. 최적 웨이포인트 선택 및 이동 명령
        target_waypoint = self._select_optimal_waypoint()
        if target_waypoint is not None:
            self.publish_move_command(self.current_pos, target_waypoint)
            
        return False  # 아직 최종 목적지 미도착

    def _should_update_route(self, current_time):
        """경로 업데이트가 필요한지 판단"""
        # 경로가 없음
        if self.current_route is None:
            return True
        
        # 일정 시간 간격으로 경로 재검증 (선택적)
        if current_time - self.last_route_update > 1.0:
            # 현재 위치에서 너무 멀리 벗어났는지 확인
            if self.route_idx < len(self.current_route):
                nearest_point = self.current_route[self.route_idx]
                if not self.distance_okay(nearest_point, multiplier=2, arrive=False):  # 멀리 벗어남
                    return True
        
        return False

    def _update_route(self, destination):
        """경로 업데이트"""
        new_route = self.a_star_search(self.current_pos, destination, return_world=True)
        new_route = new_route[1:]
        
        if not new_route:
            new_route = [np.copy(destination)]
            
        self.current_route = new_route
        self.route_idx = 0
        
        return True

    def _select_optimal_waypoint(self):
        """최적 웨이포인트 선택 - 핵심 로직"""
        if not self.current_route or self.route_idx >= len(self.current_route):
            return None
        
        # Phase 1: 조건을 만족하는 가장 가까운 점 찾기
        # (가시선 확보 + 충분한 거리)
        optimal_waypoint = None
        optimal_idx = None
        
        self.current_grid_pos = self.world_to_grid(self.current_pos)
        for i in range(self.route_idx, len(self.current_route)):
            waypoint = self.current_route[i]

            # 조건 1: 가시선이 확보되었는가?
            grid_point = self.world_to_grid(waypoint)
            if not check_line_of_sight(self.current_grid_pos, grid_point, self.grid_map):
                continue  # 가려짐, 다음 점 확인
            
            # 조건 2: 충분한 거리인가?
            if self.distance_okay(waypoint, multiplier=1):
                continue  # 너무 가까움, 다음 점 확인
                
            # 조건을 만족하는 첫 번째(가장 가까운) 점 발견
            optimal_waypoint = waypoint
            optimal_idx = i
            break
        
        # Phase 2: 조건을 만족하는 점을 찾았다면 사용
        if optimal_waypoint is not None:
            self.route_idx = optimal_idx
            return optimal_waypoint
        
        # Phase 3: 조건을 만족하는 점이 없다면 Fallback
        # 경로상 가장 가까운 점을 선택하고 인덱스 진행
        fallback_waypoint = self.current_route[self.route_idx]
        
        # 다음 웨이포인트로 진행 (단, 경로 끝이 아닌 경우만)
        if self.route_idx < len(self.current_route) - 1:
            self.route_idx += 1
        
        return fallback_waypoint

    def _reset_route(self):
        """경로 정보 초기화"""
        self.current_route = None
        self.current_destination = None
        self.route_idx = 0

    def publish_move_command(self, current_position, target_waypoint):
        """
        Publishes a command for the robot to move to the given target waypoint.
        """
        ttw = [np.round(target_waypoint[0],5), np.round(target_waypoint[1],5)]
        tcp = [np.round(current_position[0],5), np.round(current_position[1],5)]
        dx = ttw[0] - tcp[0]
        dy = ttw[1] - tcp[1]
        yaw = np.arctan2(dy, dx)

        pose2d = Pose2D()
        pose2d.x = ttw[0]
        pose2d.y = ttw[1]
        pose2d.theta = yaw

        self.pose_pub.publish(pose2d)
        # rospy.loginfo(f"[explore_node] ({self.state}) Destination ({pose_msg.x}, {pose_msg.y}) with heading {np.rad2deg(pose_msg.theta):.2f}°")
    
    def add_avoid_points(self, new_avoid_list):
        """
        주어진 3D 월드 좌표 리스트(avoid_list)를 그리드 좌표로 변환하고,
        해당 그리드 셀을 장애물(벽)로 설정합니다.
        새롭게 추가되는 점들만 이 함수를 호출해서 설정한다면 연산 시간을 아낄 수 있습니다.
        만약 현재까지 avoid_list를 초기화하고 싶다면 reset_map()을 사용해주세요.

        Args:
            avoid_list (list): 장애물로 추가할 3D 월드 좌표들의 리스트.
                                 예: [[x1, y1, z1], [x2, y2, z2], ...]
        """
        for point in new_avoid_list:
            grid_coords = self.world_to_grid(point)
            if grid_coords:
                row, col = grid_coords
                if self.grid_map[row][col] >= 1: # 1이 벽
                    continue
                self.grid_map[row][col] = 1 
                components = _find_connected_components(self.grid_map)
                if len(components) > 1: # 연결되지 못한 경우
                    self.grid_map[row][col] = 0 

    def reset_map(self):
        """
        그리드 맵을 장애물이 추가되기 전의 초기 상태로 복원합니다.
        """
        self.grid_map = np.copy(self.original_grid_map)
    
    def grid_to_world(self, grid_coords, ddd=True): # TODO: 각 점들의 중심점으로 하는 방안
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

def check_line_of_sight(start_grid, end_grid, occupancy_grid):
    """
    Bresenham's Line Algorithm을 사용하여 두 지점 사이의 시야(LOS)를 확인합니다.
    입력 좌표는 (행, 열) 순서라고 가정합니다.
    """
    # (행, 열) 순서의 입력을 내부 변수 (y, x)에 올바르게 할당합니다.
    p1 = np.array(start_grid, dtype=int)
    p2 = np.array(end_grid, dtype=int)
    
    y1, x1 = p1  # 수정된 부분: (row, col) -> y1, x1
    y2, x2 = p2  # 수정된 부분: (row, col) -> y2, x2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    height, width = occupancy_grid.shape
    
    x, y = x1, y1

    while True:
        # 현재 지점 (x, y)가 그리드 범위 내에 있는지 확인
        if not (0 <= x < width and 0 <= y < height):
            return False  # 맵을 벗어나면 장애물이 있는 것으로 간주

        # 현재 지점에 장애물이 있는지 확인 (벽 = 1)
        # NumPy 인덱싱은 [row, col] 순서이므로 [y, x]로 접근
        if occupancy_grid[y, x] == 1:
            return False

        # 목표 지점에 도달하면 루프 종료
        if x == x2 and y == y2:
            break
        
        # 다음 지점 계산
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
            
    return True

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, json, math, re

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))


import numpy as np
# -------- ROS --------
import rospy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped, Pose2D
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import defaultdict, deque

class PartialSGPlanner(object):
    """
    /partial_sg(JSON) + /target_obj_idx(Int32) → /waypoint_goal(PoseStamped)
    /state_estimation(Odometry)로 도착 판정 → 충분히 가까우면 다음 inference 대기
    """
    def __init__(self):
        # 고정 토픽
        self.partial_sg_topic = "/partial_scene_graph_generator/partial_scene_graph"
        self.llm_response = "/llm_response"
        self.odom_topic       = "/state_estimation"
        self.image            = "/camera/image/compressed"
        self.waypoint_topic   = "/way_point_with_heading"
        self.challenge_question = "challenge_question"
        self.llm_query = "/user_query"
        self.llm_image = "/user_image"
        self.llm_sg = "/partial_sg"
        self.llm_visit = "/visited_list"
        self.map_frame        = "map"

        self.dist_ths    = float(rospy.get_param("~dist_ths", 0.5))
        self.enable_logs = bool(int(rospy.get_param("~enable_logs", 1)))

        self.nodes = None
        self.a_nodes = None
        self.a_nodes_np = None
        self.a_threshold = 0.55 # a_node_size 가 0.5로 설정되어 있음

        self.traverse        = False
        self.solved          = False
        self.response_waiting = False
        self.current_goal    = None
        self.latest_partial_sg = None
        self.latest_pose     = None
        self.latest_image      = None
        self.latest_idx = None

        self.sent_partial_sg = None
        self.sent_pose = None
        self.sent_image = None

        self.graph = defaultdict(list)
        self.goal_to_move = ["north", "east", "west", "south", "north_east", "north_west", "south_east", "south_west", "random_exploration", "solved"]
        self.goal = None
        self.avoidance = None
        self.avoid_list = []

        self.route = None
        self.route_idx = None

        self.pending_idx     = None  # idx만 먼저 들어온 경우 보관

        self.visited_list = []

        self.buffer_dict = {
            'image': deque(maxlen=10),
            'partial_sg': deque(maxlen=10),
            'pose': deque(maxlen=10)
        }

        # Pub/Sub
        self.mode_pub = rospy.Publisher('/planner_mode', String, queue_size=1)
        self.goal_pub = rospy.Publisher(self.waypoint_topic, Pose2D, queue_size=10, latch=True)
        rospy.Subscriber(self.partial_sg_topic, String, self.cb_partial_sg, queue_size=1)
        rospy.Subscriber(self.llm_response, String,  self.cb_llm_response, queue_size=1)
        rospy.Subscriber(self.odom_topic,       Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber(self.image, CompressedImage, self.cb_image, queue_size = 1)
        rospy.Subscriber(self.challenge_question, String, self.cb_question, queue_size = 1)
        
        rospy.Subscriber("/node_list", PointCloud2, self.node_callback)
        rospy.Subscriber("/a_node_list", PointCloud2, self.cb_a_nodes, queue_size=1)
        rospy.Subscriber("/mst_edges_marker", Marker, self.mst_callback)

        rospy.Timer(rospy.Duration(0.2), self.timer_callback)
        self.topic_buffer = rospy.Timer(rospy.Duration(1.0), self.buffer)

        self.llm_query_pub = rospy.Publisher(self.llm_query, String, queue_size=1, latch=True)
        self.llm_image_pub = rospy.Publisher(self.llm_image, CompressedImage, queue_size=1, latch=True)
        self.llm_sg_pub = rospy.Publisher(self.llm_sg, String, queue_size=1, latch=True)
        self.llm_visit_pub = rospy.Publisher(self.llm_visit, String, queue_size=1, latch = True)

        rospy.loginfo("[Planner] ready. dist_ths=%.3f", self.dist_ths)
    
    def timer_callback(self, event):
        # important to set validation function to check whether task 3 is solved or not. --> solved --> self.solved = True
        if not self.solved:
            self.traverse_and_question()
        else:
            msg = String()
            msg.data = "task3 solved"
            self.mode_pub.publish(msg)
    
    def buffer(self, event):

        if self.latest_image and self.latest_partial_sg and self.latest_pose:
            self.buffer_dict['image'].append(self.latest_image)
            self.buffer_dict['partial_sg'].append(self.latest_partial_sg)
            self.buffer_dict['pose'].append(self.latest_pose)
        
        else:
            rospy.logwarn_throttle(5, "Waiting for full data set before buffering...")
    
    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.latest_pose = np.array([p.x, p.y, p.z])

        # for initial inquiry
        if self.current_goal is None:
            self.current_goal = self.latest_pose

    def cb_image(self, msg):
        self.latest_image = msg

    def cb_question(self, msg):
        self.challenge_question = msg.data
    
    def node_callback(self, msg):
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
    
    def cb_a_nodes(self, msg: PointCloud2):
        a_nodes = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        self.a_nodes = a_nodes
        self.a_nodes_np = np.array(a_nodes, dtype=float) if len(a_nodes) > 0 else None
        # degree 기반 패널티 사전 계산
        self.compute_a_node_degree_penalty()

        if self.a_nodes_np is not None:
            rospy.loginfo("[action_node] received %d a_nodes", len(self.a_nodes_np))

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


    def cb_partial_sg(self, msg: String):
        try:
            self.last_partial_sg = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("[Planner] bad partial_sg JSON: %s", e)
            return

    def cb_llm_response(self, msg: String):
        # llm string 답변을 parsing해서 goal 값과 avoid 값을 분리
        response_str = msg.data
        rospy.loginfo(f'[Planner] LLM response as: "{response_str}"')
        try:
            data = json.loads(response_str)
            if not isinstance(data, dict):
                self.get_logger().error(f"no dict (type: {type(data)})")
                return
            
            self.goal = data.get("goal")
            self.avoidance = data.get("avoidance")
            
            rospy.loginfo(f'[Planner] parsing success: Goal="{self.goal}", Avoidance={self.avoidance}')

        except json.JSONDecodeError:
            rospy.logwarn(f"[Planner] No json: {response_str}")
        except Exception as e:
            rospy.logwarn(f"[Planner] exception in LLM callback as : {e}")

        new_avoid_list = self.avoid_append(self.avoidance)
        # avoid 리스트에 새로운 좌표가 생겼는지 여부 판단
        if self.avoid_list != new_avoid_list :
            new_avoid = True
        else:
            new_avoid = False
        self.avoid_list = new_avoid_list

        self.current_goal = self.handle_goal_with_direction(self.goal, avoid_go = new_avoid)
        
        self.response_waiting = False
    
    # goal 의 경우 방향 대로 현재 지점 기준 목표 좌표 계산
    # 1m 내외? 최소 20cm 보다는 크게 방향 벡터를 설정(a stor 그리드 사이즈보다는 크도록)
    # 만약 random traverse가 나왔다면, MST의 다음 node의 좌표를 반환 (그대로 node를 반환할 것이냐 아니면 node 방향 기준으로 중간 목표점을 설정할 것이냐)
    def handle_goal_with_direction(self, direction: String, avoid_go : bool, step_size = 1.0):
        if direction not in self.goal_to_move :
            rospy.logwarn(f"[Planner] Invalid direction type in llm response as {direction}")
            return None
        
        # 모든 프로세스 종료.
        if direction == "solved":
            self.solved = True
            return self.latest_pose
            # timer에서 바로 종료 시그널 출력하도록 함
        
        # 편집된 mst에서 최인접 노드 반환
        if direction == "random_exploration":
            goal = self.mst_random_exploration(avoid_go)
            return goal
        
        root2 = 2 ** 0.5
        direction_to_vector = {
            "north" : [1,0], 
            "east" : [0,-1], 
            "west": [0, 1], 
            "south": [-1, 0], 
            "north_east": [1/root2, -1/root2], 
            "north_west": [1/root2, 1/root2],
            "south_east": [-1/root2, -1/root2], 
            "south_west": [-1/root2, 1/root2], 
        }

        # step size로 방향 벡터 크기 조절
        vector_to_move = step_size * np.array(direction_to_vector[direction]) if direction in direction_to_vector else np.array([0, 0])

        # latest_pose가 아니라, 질문 보낸 시점의 sent_pose 값을 사용함
        current_position = self.sent_pose.pose.position
        current_orientation = self.sent_pose.pose.orientation

        q = [current_orientation.x, 
        current_orientation.y, 
        current_orientation.z, 
        current_orientation.w]
        
        (roll, pitch, yaw) = euler_from_quaternion(q)
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        map_offset = rotation_matrix.dot(vector_to_move)
        goal_vector = np.array([
            current_position.x + map_offset[0],
            current_position.y + map_offset[1],
            current_position.z
        ])
        return goal_vector

    def avoid_append(self, avoidance, dist_ths = 0.3):
        # avoid할 좌표들을 리스트에 계속 추가 (set으로 중복 방지)
        # MST에 avoid 노드 고려해 편집하는 함수도 추가
        if not isinstance(avoidance, list):
            rospy.logwarn("[Planner] invalid type avoidance", avoidance)
        
        psg = self.sent_partial_sg["objects"]

        # case 1) near
        if len(avoidance) == 1:
            obj_idx = str(avoidance[0])
            obj = next((obj for obj in psg if obj.get("object_id") == obj_idx), None)

            if obj is None:
                rospy.logwarn(f"[Planner] invalid bbox center for idxes {obj_idx}")
                return
            
            # ensure that scene graph objects have 'center' attribute
            obj_point = np.array(obj.get("center"))

            if not self.avoid_list:
                self.avoid_list.append(obj_point)
            else:
                avoid_points = np.array(self.avoid_list)
                diff_vectors = avoid_points - obj_point
                distances = np.linalg.norm(diff_vectors, axis=1)
                is_too_close = np.any(distances < dist_ths)
                if not is_too_close:
                    self.avoid_list.append(obj_point)

        # case 2) between
        if len(avoidance) == 2:
            obj_idx_1 = str(avoidance[0])
            obj_1 = next((obj for obj in psg if obj.get("object_id") == obj_idx_1), None)

            obj_idx_2 = str(avoidance[1])
            obj_2 = next((obj for obj in psg if obj.get("object_id") == obj_idx_2), None)

            if obj_1 is None or obj_2 is None:
                rospy.logwarn(f"[Planner] invalid bbox center for idxes {obj_idx_1}, {obj_idx_2}")
                return
            
            # ensure that scene graph objects have 'center' attribute
            obj_point_1 = np.array(obj_1.get("center"))
            obj_point_2 = np.array(obj_2.get("center"))
            
            obj_between = np.mean([obj_point_1, obj_point_2], axis=0)

            if not self.avoid_list:
                self.avoid_list.append(obj_between)
            else:
                avoid_points = np.array(self.avoid_list)
                diff_vectors = avoid_points - obj_between
                distances = np.linalg.norm(diff_vectors, axis=1)
                is_too_close = np.any(distances < dist_ths)
                if not is_too_close:
                    self.avoid_list.append(obj_between)

    def mst_random_exploration(self, avoid_go:bool, dist_ths = 0.5):
        # 기존의 MST 내에서 탐색하는 코드 모두 버리고, 그냥 MST 편집하는 코드만 유지
        # MST 편집 후 가장 가까운 노드를 지정해서 이동하도록 함

        # self.graph의 각 node에 대해서 avoid 좌표 고려해 제외만...
        # avoid list가 달라졌을 경우만 진행하도록 + 제외 로직 비용 절감 
        # node_list 자체가 pc2 좌표 이므로, 해당 list에서 뽑으면 될 듯

        if not self.graph or not self.avoid_list:
            rospy.logwarn("[Planner] graph or avoid list empty; cannot reapply MST.")
            return False
        
        # 1) 회피 노드 제거한 인접 리스트 구성 및 최인접 노드 반환
        if avoid_go:
            pruned = defaultdict(list)
            except_node_list  = [exp_node for exp_node in self.graph.keys() if self.avoid_list and np.linalg.norm(np.array([p[:2] for p in self.avoid_list]) - np.array(self.nodes[exp_node][:2]), axis=1).min() < dist_ths]
            
            for u, nbrs in self.graph.items():
                if u in except_node_list:
                    continue
                for v in nbrs:
                    if v in except_node_list:
                        continue
                    pruned[u].append(v)
            
            self.graph = pruned
        
        candidates = list(self.graph.keys())
        start = min(candidates, key=lambda n: np.linalg.norm(np.array(self.nodes[n][:2]) - self.latest_pose[:2]))
        mst_goal_node = np.array(self.nodes[start])
        return mst_goal_node
    
    def _distance_okay_xy(self, latest_pose, waypoint, dist_ths):
        return (np.linalg.norm(np.array(latest_pose[:2]) - np.array(waypoint[:2])) < dist_ths)

    def traverse_and_question(self):
        if self.latest_pose is None or self.current_goal is None :
            rospy.logwarn_throttle(1.0, "[Planner] No /state_estimation observed")
            return
        
        # LLM에 질문을 보낼 것인지, waypoint를 발행할 것인지 결정
        dist_to_goal = np.linalg.norm(self.latest_pose[:2] - self.current_goal[:2])
        if dist_to_goal < self.dist_ths : 
            self.traverse = False
        else:
            self.traverse = True
        
        # self.response_waiting 으로 LLM 답변올 때까지 전체 프로세스 대기
        # self.response_waiting 는 LLM callback이 성공적으로 모두 진행되었을 경우 True로 바뀜
        # 성공적으로 바뀌었다면, current_goal이 정상적으로 바뀌어, self.traverse = True로 되어 waypoint 발행하게 됨
        if not self.traverse and not self.response_waiting:
            self.question()
            self.response_waiting = True
        
        # 1) a star 에 avoid penalty -> 일단 가중치는 임의로 초기화. 실험하면서 조정해야 함.
        # 2) a star node list에 노드 제외 -> 비용은 좀 더 들지만 확실하긴 함. 제외할 반경 설정해야 함.
        # 1) 과 2) 합쳐서 구현함
        if self.traverse and not self.response_waiting :

            # self.route 가 None일때 a_star_compute가 한번 실행돼서 rotue를 생성. 이후 route 대로 따라
            if self.route is None :
                start_xy = np.array(self.latest_pose[:2], dtype=float)
                goal_xy  = np.array(self.current_goal[:2], dtype=float)

                # a-node에서 가장 가까운 start, goal 선택
                start_idx = np.argmin(np.linalg.norm(self.a_nodes_np[:, :2] - start_xy, axis=1))
                goal_idx  = np.argmin(np.linalg.norm(self.a_nodes_np[:, :2] - goal_xy, axis=1))

                start = tuple(self.a_nodes_np[start_idx])
                goal  = tuple(self.a_nodes_np[goal_idx])

                total_cost, path = self.a_star_compute(start, goal)
                if not np.isfinite(total_cost) or len(path) == 0:
                    rospy.logwarn("[action_node] A* failed: no path")
                    return

                # 결과 저장 및 디버그 출력
                self.route = path
                self.route_idx = 0
                
                rospy.loginfo("[action_node] planned path: %d waypoints, cost=%.3f", len(self.route), total_cost)
            

            # a star route를 완주했을 경우, visited list 추가 + route 초기화
            if self.route_idx >= len(self.route):
                self.visited_list_append()
                self.route = None
                self.traverse = False   # 사실 위의 거리 기반 판단 함수로 다시 지정되긴 하지만, 이해를 돕기 위해 포함함
                return
            
            waypoint = self.route[self.route_idx]
            if not self._distance_okay_xy(self.latest_pose, waypoint, self.dist_ths):
                # 목표 웨이포인트로 향하는 yaw 계산
                dx = waypoint[0] - self.latest_pose[0]
                dy = waypoint[1] - self.latest_pose[1]
                yaw = math.atan2(dy, dx)

                cmd = Pose2D()
                cmd.x = float(waypoint[0])
                cmd.y = float(waypoint[1])
                cmd.theta = float(yaw)
                self.goal_pub.publish(cmd)
            else:
                # 다음 웨이포인트로
                self.route_idx += 1
                rospy.loginfo(f"[Planner] waypoint reached. moving next waypoint as {self.route_idx} in total {len(self.route)} nodes]")
        
        if self.response_waiting :
            rospy.loginfo_throttle(1.0, "[Planner] waiting for llm response ...")
    
    def visited_list_append(self):
        if self.last_partial_sg is None or self.latest_pose is None :
            rospy.logwarn("[Planner] No SG or No pose")
            return
        cur_pose = self.latest_pose

        # sg 형식 통일해야 함!
        # 목표 지점 도달했을 시, 거리 기반으로 visited object label을 추가하는 간단한 방식이라, 
        # 실제로 의미있는 오브젝트 라벨만 넣거나, 중복 또는 순서 고려해 보완할 필요 있음.
        objects = self.last_partial_sg.get("objects")
        for obj in objects:
            obj_center = obj.get("center")
            if obj_center is not None:
                obj_center = np.array(obj_center)
            else:
                continue
            if self._distance_okay_xy(cur_pose, obj_center, 0.2):
                self.visited_list.append(obj.get("raw_label"))

    def a_star_compute(self, start, goal):
        """A*: 스텝 비용 및 휴리스틱에 degree/erase 패널티를 포함 (기존 로직과 동일 철학)."""
        a_nodes = self.a_nodes
        if not a_nodes:
            return float('inf'), []

        # open/closed sets
        open_set = {start}
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        visited = set()

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                total_cost = sum(self.heuristic(path[i], path[i+1]) for i in range(len(path)-1))
                return total_cost, path

            open_set.remove(current)
            visited.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal) + \
                                        0.3 * self.direction_change_penalty(current, neighbor, came_from)
                    open_set.add(neighbor)

        return float('inf'), []
    
    # !!! A star  codes !!!
    # ----------------------------------------------------------------------------------------
    def get_neighbors(self, current):
        """치비셰프 거리 기반 이웃 (격자 간격 ~ a_star_node_size)."""
        if self.a_nodes is None:
            return []
        neighbors = []
        c = np.array(current[:2], dtype=float)
        for n in self.a_nodes:
            if tuple(n) == current:
                continue
            diff = np.abs(c - np.array(n[:2], dtype=float))
            if max(diff[0], diff[1]) <= self.a_threshold:
                neighbors.append(tuple(n))
        return neighbors
    
    def compute_a_node_degree_penalty(self):
        """각 a-node의 연결 차수 기반 패널티(차수가 낮을수록 패널티↑)."""
        if self.a_nodes is None or len(self.a_nodes) == 0:
            self.a_nodes_penalty = None
            return

        penalties = []
        for node in self.a_nodes:
            neighbors = self.get_neighbors(tuple(node))
            if len(neighbors) == 0:
                penalties.append(float('inf'))
            else:
                penalties.append(4.0 / len(neighbors))  # 8방향 완전연결이면 0.5
        self.a_nodes_penalty = penalties
    
    # 여기에 그냥 일정 반경 안에 self.avoid_list 있으면 그냥 해당 노드 고려 하지 않음 --> 결국 제거 효과와 동일
    def erase_penalty(self, p_xyz_tuple, erase_penalty_scale=2.0, erase_penalty_radius = 1.5):
        """
        erase_node 회피 패널티 (inverse-distance).
        반경 R 내에서: scale * (1/max(d, eps) - 1/R), 반경 밖: 0
        """
        if self.avoid_list is None or erase_penalty_scale <= 0.0 or erase_penalty_radius <= 0.0:
            return 0.0
        p = np.array(p_xyz_tuple[:2], dtype=float)
        diffs = np.array([p[:2] for p in self.avoid_list]) - p
        dists = np.linalg.norm(diffs, axis=1)
        if dists.size == 0:
            return 0.0
        min_d = float(dists.min())
        if min_d < 1.0 :
            return 1e3
        
        if not np.isfinite(min_d) or min_d >= erase_penalty_radius:
            return 0.0
        eps = 1e-3
        return erase_penalty_scale * max(1.0 / max(min_d, eps) - 1.0 / erase_penalty_radius, 0)

    def heuristic(self, p1, p2):
        """
        휴리스틱(겸 스텝 비용): 1.5*거리 + degree penalty(p1) + erase penalty(p1)
        (정확 최단 보장보다 경로 성향/회피를 강조하는 설계)
        """
        base = 1.5 * np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))
        idx = self._a_node_index_of_tuple(p1)
        degree_pen = self.a_nodes_penalty[idx] if (self.a_nodes_penalty is not None and idx is not None) else 0.0
        erase_pen  = self.erase_penalty(p1)
        return base + degree_pen + erase_pen

    def direction_change_penalty(self, current, neighbor, came_from):
        prev = came_from.get(current)
        if prev is None:
            return 0.0
        v1 = np.array(current[:2]) - np.array(prev[:2])
        v2 = np.array(neighbor[:2]) - np.array(current[:2])
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.dot(v1, v2) / denom
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    def _a_node_index_of_tuple(self, node_tup):
            """tuple을 self.a_nodes에서 찾아 index 반환 (동일 소스에서 생성된 튜플이라면 == 비교로 충분)."""
            if self.a_nodes is None:
                return None
            for i, n in enumerate(self.a_nodes):
                if tuple(n) == node_tup:
                    return i
            return None

    def question(self):
        if not (self.buffer_dict['image'] and self.buffer_dict['partial_sg'] and self.buffer_dict['pose']):
            rospy.logwarn("question(): buffer_dict is empty")
            return

        latest_image = self.buffer_dict['image'][-1] if self.buffer_dict['image'] else None
        latest_partial_sg = self.buffer_dict['partial_sg'][-1] if self.buffer_dict['partial_sg'] else {}
        latest_pose = self.buffer_dict['pose'][-1] if self.buffer_dict['pose'] else None
        
        self.sent_partial_sg = latest_partial_sg
        self.sent_pose = latest_pose
        self.sent_image = latest_image

        if latest_image is not None:
            self.llm_image_pub.publish(latest_image)

        json_ready = self.visited_list   #[[label, list(coord)] for label, coord in self.visited_list]
        json_str = json.dumps(json_ready, indent=2)
        
        prompt = self.build_prompt(
                question=self.challenge_question,
                partial_sg=latest_partial_sg,
                guidance= "GOAL SELECTION : You have to solve question with given panorama image (width = w*(yaw + pi)/2*pi, So w equals zero or ends of image is same as south). " \
                "You have to answer only with the given word in 'GOAL_TO_MOVE', only 1 word can be chosen. You have to consider already visited points in VISITED_LIST" \
                "If question is solved, than return 'solved' in GOAL_TO_MOVE. If logical movenment is not possible, than return 'random_exploration' in GOAL_TO_MOVE." \
                "AVOIDANCE : You have to return the object index to avoid from the given constraints in the question IF corresponding objects exist in SCENE INFORMATION. You have to select 1 or 2 object index to avoid in SCENE INFORMATION - OBJECTS. " \
                "If constraint is to avoid the space 'near' of the object, than return the 1 index of corresponding object" \
                "If constraint is to avoid the space 'between' the objects, than return the 2 index of corresponding objects",
                visited_list=String(json_str),
                # 만약 random explore가 나왔다면, mst_random_exploration 대로 진행.
                goal_to_move=self.goal_to_move,
                output_format=(
                    "OUTPUT FORMAT (STRICT):\n"
                    "- Respond with following json format.\n"  
                    "- Respond the chosen goal at 'goal' and index of object to avoid at 'avoidance'\n"
                    "- Ex: {'goal':'south-west', 'avoidance':[0]}, {'goal':'random_exploration', 'avoidance': [2, 4]}, {'goal':'west', 'avoidance': None}"
                )
        )

        msg_query = String()
        msg_query.data = prompt
        self.llm_query_pub.publish(msg_query)

        rospy.loginfo("[LeoPlanner] Published query, image, sg to LLM.")
    
    def build_prompt(self, question: str, partial_sg: dict, guidance: str, visited_list: str, goal_to_move:str, output_format: str) -> str:
        scene = partial_sg.get("scene_name", "Unknown") # scene_name 추출
        nodes = partial_sg.get("objects", []) or []
        edges = partial_sg.get("relationships", []) or []

        # object idx를 주어서, avoidance 여부를 판단하는 목적
        # relation의 경우 object list를 사용할 시 제외해야 함
        obj_lines = [f"- object {n.get('object_id')}: {n.get('raw_label')}" for n in nodes]
        rel_lines = []
        for e in edges:
            try:
                # e = [src, predicate, dst] 가정
                rel_lines.append(f"- object {e[0]} is {e[1]} object {e[2]}")
            except (IndexError, TypeError):
                pass
        
        objects_str = chr(10).join(obj_lines)
        relationships_str = chr(10).join(rel_lines)

        prompt = f"""\
            SYSTEM:
            You are an AI assistant helping a robot understand a 3D scene.
            You will be given a scene graph with a list of objects (with indices) and their relationships.
            Your task is to analyze the scene graph and the user's question to provide a precise answer in the required format.

            SCENE INFORMATION:
            - Scene: {scene}
            - OBJECTS:
            {objects_str}
            - RELATIONSHIPS:
            {relationships_str}

            VISITED LIST:
            {visited_list}

            GOAL_TO_MOVE:
            {goal_to_move}
            
            USER REQUEST:
            "{question}"

            TASK:
            {guidance}

            {output_format}
            """
        return prompt.strip()


def main():
    rospy.init_node("planner_partialSG", anonymous=False)
    node = PartialSGPlanner()
    rospy.loginfo("planner_partialSG node started.")
    rospy.spin()


if __name__ == "__main__":
    main()
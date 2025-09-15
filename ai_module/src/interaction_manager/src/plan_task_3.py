#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys, os, json, math, re
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))

import numpy as np
# -------- ROS --------
import rospy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String, Int32, Bool
from geometry_msgs.msg import PoseStamped, Pose2D
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import defaultdict, deque
from planning_node.a_star_algorithm import A_star
import time
from openai import OpenAI
import base64, cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as rot

class random_exploration:
    def __init__(self, graph, nodes, dist_ths):
        self.graph = graph
        self.nodes = nodes
        self.traversal_order = []
        self.traversal_simple_order = [] 
        self.visited_mst_list = []
        self.latest_avoid_list = None
        self.dist_ths = dist_ths
    
    def logger(self, txt, type="log"):
        try:
            if type == "log":
                rospy.loginfo(txt)
            elif type == "warn":
                rospy.logwarn(txt)
            else:
                pass
        except:
            print(txt)
        
    def explore_with_mst(self, pruned, latest_pose): # avoid node, visited node는 제외하고 순회 순서를 구하도록 알고리즘 수정
        if self.graph is None or self.nodes is None or latest_pose is None:
            self.logger(f"[Random_Explorer] explore_with_mst(): Error at processing data, please check data as graph : {self.graph}, nodes: {self.nodes}, latest_pose : {latest_pose}", "warn")
            return
        
        if len(self.graph) < 1 :
            self.logger("[Random_Explorer] explore_with_mst(): pruned MST empty.")
            self.traversal_order = []
            self.traversal_simple_order = []
            return
        
        #pruned = self.graph
        
        max_degree = max(len(pruned[n]) for n in pruned)
        candidate_roots = [n for n in pruned if len(pruned[n]) == max_degree]

        def distance(n1, n2): 
            return np.linalg.norm(np.array(n1[:2]) - np.array(n2[:2]))
        
        root = min(candidate_roots, key=lambda n: distance(self.nodes[n], latest_pose))
        
        def dfs_traversal(graph, start_node):
            visited = set()
            traversal_order = []
            
            def dfs(node):
                visited.add(node) 
                traversal_order.append(node) 
                for neighbor in graph[node]: 
                    if neighbor not in visited: 
                        dfs(neighbor) 
                        traversal_order.append(node) # backtracking

            dfs(start_node)
            return traversal_order

        self.traversal_order = dfs_traversal(pruned, root)

    def mst_random_exploration(self, avoid_list, latest_pose):
        if len(self.graph) < 1:
            self.logger(f"[Random_Explorer] mst_random_exploration(): Error at processing data, please check data as graph : {self.graph}", "warn")
            return False
        
        if avoid_list is None:
            self.logger(f"[Random_Explorer] mst_random_exploration(): Error at processing data, please check data as avoid_list : {avoid_list}", "warn")
            return False
        
        avoid_go=True
        # 이전 avoid_list와 다른지 확인
        if len(avoid_list)==0:
            avoid_go=False
        
        self.latest_avoid_list = avoid_list

        # 1) 회피 노드 제거한 인접 리스트 구성 및 최인접 노드 반환
        pruned = defaultdict(list)
        except_node_list = set(self.visited_mst_list)

        if avoid_go:
            nodes_to_avoid={exp_node for exp_node in self.graph.keys() if np.linalg.norm(np.array([p[:2] for p in avoid_list]) - np.array(self.nodes[exp_node][:2]), axis=1).min() < self.dist_ths}
            except_node_list.update(nodes_to_avoid)

        for u, nbrs in self.graph.items():
                if u in except_node_list:
                    continue
                for v in nbrs:
                    if v in except_node_list:
                        continue
                    pruned[u].append(v)
            
        #self.graph = pruned
        candidates = list(pruned.keys())

        if not candidates:
            self.logger(f"[Random_Explorer] No valid MST candidates... candidates : {candidates}", "warn")
            x, y, z = latest_pose
            return latest_pose  #np.array([x+1, y, z])
        
        sorted_candidates = sorted(
            candidates, 
            key=lambda n: np.linalg.norm(np.array(self.nodes[n][:2]) - latest_pose[:2])
        )
        
        # default option) 가장 가까이 있는 노드가 아닌, 미리 빌드해둔 MST를 따라 순회하기
        goal_node_idx = None 
        nearest_node = None
        nearest_dist = 1e9 
        nearest_ths_node = None 
        for n_idx in sorted_candidates:
            dist_to_node = np.linalg.norm(np.array(self.nodes[n_idx][:2]) - latest_pose[:2]) 
            if dist_to_node < nearest_dist: 
                nearest_node = n_idx 
                nearest_dist = dist_to_node 
            if dist_to_node > self.dist_ths: 
                nearest_ths_node = n_idx

        # 가장 가까운 노드가 충분히 멀리 있으면 (self.dist_ths 보다 멀리 있음) 그대로 사용
        if nearest_node == nearest_ths_node or nearest_ths_node is None:
            goal_node_idx = nearest_node
        else:
            # MST 순회 로직 사용
            # self.traversal_order에서 nearest_node '다음' 미방문 노드를 찾는다. 
            # 순서가 없거나 오래되었을 수 있으니 필요 시 재구성
            if not getattr(self, "traversal_order", None):
                self.explore_with_mst(pruned, latest_pose)
                rospy.loginfo("[Planner] Build MST traversal order")
            
            self.logger(f"[Random_Explorer] traversal_order length: {len(self.traversal_order) if self.traversal_order else 0}")
            self.logger(f"[Random_Explorer] visited_mst_list: {self.visited_mst_list}")
            
            next_idx = None
            # 1) 현재 traversal_order에서 nearest_node의 다음 미방문 노드 탐색 
            try:
                self.logger("[Random_Explorer] Finding next unvisited node")
                pos = self.traversal_order.index(nearest_node) 
                for k in range(pos + 1, len(self.traversal_order)): 
                    cand = self.traversal_order[k]
                    if cand not in self.visited_mst_list and cand in pruned:
                        next_idx = cand
                        break
            
            except ValueError:
                    pass
                
            # 2) 그래도 못 찾으면: 후보들 중(가까운 순) 미방문 첫 노드로 폴백
            if next_idx is None:
                for cand in sorted_candidates:
                    if cand not in self.visited_mst_list:
                        next_idx = cand
                        break
                # 모든 후보가 방문됨 → 마지막 폴백: nearest_node 자체로 진행
                if next_idx is None:
                    next_idx = nearest_node

            goal_node_idx = next_idx
            self.logger(f"[Random_Explorer] MST goal node assigned as {goal_node_idx}")

        # goal_node_idx가 여전히 None인 경우 처리
        if goal_node_idx is None:
            self.logger(f"[Random_Explorer] All remaining MST nodes are too close to the current position.", "warn")
            return latest_pose #np.array([x+1, y, z])

        self.visited_mst_list.append(goal_node_idx)
        mst_goal_node = np.array(self.nodes[goal_node_idx])
        self.logger(f"[Random_Explorer] goal as {goal_node_idx} : {mst_goal_node}")
        return mst_goal_node

class plan_task_3(object):
    def __init__(self):
        self.challenge_question=None
        api_key = rospy.get_param("~openai_api_key", os.environ.get("OPENAI_API_KEY"))
        self.client = OpenAI(api_key=api_key)
        self.model="gpt-4o"
        self.wait_for_sg=False
        self.last_partial_sg=None
        self.last_partial_sg_rich=None
        self.can_solve=False
        self.visited_list=[]
        self.a_star=None
        self.grid_map = None

        self.latest_image_b64=None
        self.latest_image=None
        self.bridge = CvBridge()
        self.latest_cv_img=None

        self.latest_ori=None
        self.latest_pose=None
        self.avoid_processed=False
        self.random_exploration=None
        #random exploration에 필요한 변수들
        self.graph = defaultdict(list)
        self.nodes = None
        self.dist_ths=0.45
        self.false_avoid=set()

        self.waypoint=[]
        self.between=[]
        self.avoid=[]
        self.sequence=[]

        #구독!!
        rospy.Subscriber("/challenge_question",String,self.cb_challenge_question,queue_size=1)
        rospy.Subscriber("/fusion/scene_graph_merged",String,self.cb_partial_sg,queue_size=1)
        rospy.Subscriber("/fusion/scene_graph_rich_merged",String,self.cb_partial_sg_rich,queue_size=1)
        rospy.Subscriber("/state_estimation",Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber("/grid_map_info", OccupancyGrid, self.grid_map_callback,queue_size=1)
        rospy.Subscriber("/mst_edges_marker", Marker, self.mst_callback,queue_size=1)
        rospy.Subscriber("/node_list", PointCloud2, self.node_callback)
        rospy.Subscriber("/image_scenegraph",Image,self.cb_image_sg,queue_size=1)
        
        #발행!!
        self.pub_sg=rospy.Publisher("/publish_scene_graph",Bool,queue_size=1)
        self.shutdown=rospy.Publisher("/shutdown",Bool,queue_size=1)
        self.way=rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=1)
        
    def cb_image_sg(self,msg:Image):
        self.latest_image=msg.data
        self.latest_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def cb_challenge_question(self,msg:String):
        if self.challenge_question==None:
            self.challenge_question=msg.data
    
    def node_callback(self, msg):
        try:
            if self.nodes is None:
                self.nodes = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        except Exception as e:
            rospy.logwarn(f"[Planner] nodes callback error as {e}")

    def image_coordinate(self,x,y,z):
        cam_offset=np.array([0.0, 0.0, 0.235], dtype=float)
        #점을 먼저 상대좌표로
        rel_pos=np.array([[x,y,z]],dtype=float)-np.asarray(self.latest_pose,dtype=float)
        r=rot.from_quat(self.latest_ori)
        R=r.as_dcm()
        xyz=rel_pos@R
        xyz=xyz-cam_offset
        W, H = 1920,640
        horiDis = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2) + 1e-9
        u = (W / (2 * np.pi)*np.arctan2(-xyz[:,1], xyz[:, 0])+W/2)
        v = (H / (2 * np.pi /3)*np.arctan(-xyz[:, 2] / (horiDis))+H/2)
        return int(u),int(v)

    def node_idx(self, node):
        if self.nodes is None:
            return None
        node_tuple = (round(node.x, 3), round(node.y, 3), round(node.z, 3))
        for i, n in enumerate(self.nodes):
            n_tuple = (round(n[0], 3), round(n[1], 3), round(n[2], 3))
            if node_tuple == n_tuple:
                return i
        rospy.logwarn(f"[WARNING] Node {node_tuple} not found in self.nodes.")
        return None

    def mst_callback(self, msg):
        if msg.type != Marker.LINE_LIST:
            rospy.logwarn("Received non LINE_LIST marker")
            return

        if len(msg.points) % 2 != 0:
            rospy.logwarn("LINE_LIST marker does not contain an even number of points")
            return
        
        if len(msg.points)>0:
            self.graph=defaultdict(list)
            for i in range(0,len(msg.points),2):
                p1=msg.points[i]
                p2=msg.points[i+1]
                idx1=self.node_idx(p1)
                idx2=self.node_idx(p2)
                if idx1 is not None and idx2 is not None:
                    self.graph[idx1].append(idx2)
                    self.graph[idx2].append(idx1)
        else:
            rospy.logwarn("[Planner] Received empty MST message . Keeping previous graph")

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        ori=msg.pose.pose.orientation
        self.latest_pose=np.array([p.x,p.y,p.z],dtype=float)
        self.latest_ori=np.array([ori.x,ori.y,ori.z,ori.w],dtype=float)
    
    def grid_map_callback(self, msg):
        if self.grid_map is None:
            grid_size = round(msg.info.resolution, 2)
            origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            height = msg.info.height
            width = msg.info.width
            grid_data_1d = np.array(msg.data, dtype=np.int8)
            grid_data_2d = grid_data_1d.reshape((height, width))
            self.grid_map = np.ones((height, width), dtype=np.uint8)
            self.grid_map[grid_data_2d == 0] = 0
            row_indices, col_indices = np.where(self.grid_map == 0)
            traversable_cells = list(zip(row_indices, col_indices))
            total_free_cells = len(traversable_cells)
            self.a_star = A_star(grid_size, origin, self.grid_map,rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=1))

    def cb_partial_sg(self,msg:String):
        partial_sg = json.loads(msg.data)
        out = {"objects": [], "relationships": []}
        nodes = (partial_sg or {}).get("objects", []) or []
        edges = (partial_sg or {}).get("relations", []) or []
        for node in nodes:
            out["objects"].append({
                "object_id": node.get("id"),
                "raw_label": node.get("label"),
                "center": node.get("center"),
            })
        self.last_partial_sg=out
        self.wait_for_sg=False

    def cb_partial_sg_rich(self,msg:String):
        partial_sg = json.loads(msg.data)
        regions=partial_sg.get("regions") or {}
        out_1 = {"objects": []}
        for node in regions.values():
            nodes=node.get("objects", []) or []
            for item in nodes:
                out_1["objects"].append({
                    "object_id": item.get("object_id"),
                    "raw_label": item.get("raw_label"),
                    "center": item.get("center"),
                    "bbox":item.get("bbox")
                })
        self.last_partial_sg_rich=out_1
    
    def create_avoid_path_between_objects(self, obj1_center, obj2_center, spacing=0.1):
        """ Task 3에서의 avoid 조건을 위한 함수. 두 물체 사이 직선상의 점들 리스트를 리턴함.
        """
        avoid_points = []
        x1, y1 = obj1_center[0], obj1_center[1]
        x2, y2 = obj2_center[0], obj2_center[1]
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if distance == 0:
            return [obj1_center]
        
        dx = (x2-x1)/distance
        dy = (y2-y1)/distance
        num_points = int(distance/spacing) + 1
        for i in range(num_points):
            x = x1 + i*dx*spacing
            y = y1 + i*dy*spacing
            z = obj1_center[2]
            avoid_points.append([x,y,z])
        return avoid_points

    def create_avoid_near_objects(self,bbox,threshold=0.1):
        """ Task 3에서의 avoid 조건을 위한 함수. 두 물체 사이 직선상의 점들 리스트를 리턴함.
        """
        avoid_points = []
        x_min=1e9;y_min=1e9
        x_max=-1e9;y_max=-1e9
        for i in bbox:
            x=i[0]
            y=i[1]
            if x<x_min:
                x_min=x
            if y<y_min:    
                y_min=y
            if x>x_max:
                x_max=x
            if y>y_max:
                y_max=y
        x_min-=threshold
        x_max+=threshold
        y_min-=threshold
        y_max+=threshold
        #이미 접근한 상태라 이미 늦음
        if (x_min<=self.latest_pose[0]<= x_max) and (y_min<=self.latest_pose[1]<= y_max):
            rospy.logwarn(f"Late for avoid {self.avoid}")
            return None
        avoid_points.extend(self.create_avoid_path_between_objects([x_min,y_min,0],[x_min,y_max,0],0.1))
        avoid_points.extend(self.create_avoid_path_between_objects([x_min,y_min,0],[x_max,y_min,0],0.1))
        avoid_points.extend(self.create_avoid_path_between_objects([x_max,y_max,0],[x_min,y_max,0],0.1))
        avoid_points.extend(self.create_avoid_path_between_objects([x_max,y_max,0],[x_max,y_min,0],0.1))
        return avoid_points

    def _maybe_process_avoid_once(self):
        rospy.loginfo(f"[task3] _maybe_process_avoid_once")
        """PSG, A* 준비가 되었고 avoid가 2개면 딱 한 번만 회피선을 그려준다."""
        if (self.avoid_processed or self.a_star is None or self.last_partial_sg is None or self.last_partial_sg_rich is None):
            return
        if len(self.avoid) == 1:
            #avoid near 문제
            objs=(self.last_partial_sg_rich).get("objects")
            norm =self.avoid[0].lower().replace(" ", "_")
            rospy.loginfo(f"[task3] Processing avoid: {self.avoid}")
            for obj in objs:
                if obj.get("object_id") not in self.false_avoid:   
                    raw = obj.get("raw_label").lower()
                    if raw == norm:
                        bbox=obj.get("bbox", [])
                        x_min=1e9;y_min=1e9
                        x_max=-1e9;y_max=-1e9
                        for i, (x, y, z) in enumerate(bbox):
                            x,y=self.image_coordinate(x,y,z)
                            if x_min>x:
                                x_min=x
                            if y_min>y:
                                y_min=y
                            if x_max<x:
                                x_max=x
                            if y_max<y:
                                y_max=y
                        padding=5
                        x_min-=padding;y_min-=padding
                        x_max+=padding;y_max+=padding
                        W=1920;H=640
                        x_min = max(0,min(W-1, x_min))
                        x_max = max(0,min(W-1, x_max))
                        y_min = max(0,min(H-1, y_min))
                        y_max = max(0,min(H-1, y_max))
                        quad = np.array([
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max],
                        ], dtype=np.int32)
                        vis = self.latest_cv_img.copy()
                        cv2.polylines(vis, [quad], isClosed=True, color=(255,0,0), thickness=2,lineType=cv2.LINE_AA)
                        ok, buf=cv2.imencode(".jpg", vis,[int(cv2.IMWRITE_JPEG_QUALITY),90])
                        save_dir = os.path.join(FILE_DIR, "out")
                        os.makedirs(save_dir, exist_ok=True)
                        cv2.imwrite(f"{save_dir}/avoid.jpg", vis)
                        jpeg_bytes=buf.tobytes()
                        self.latest_image_b64=base64.b64encode(jpeg_bytes).decode("ascii")
                        # To do : 이미지로 satisfy 조건 확인
                        _prompt_avoid=f"""According to the image, is the object inside the blue box is {self.avoid[0]}? 
                        Reply with exactly one token: True or False.
                        """
                        payload=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": _prompt_avoid},
                                    {"type": "input_image","image_url": f"data:image/jpg;base64,{self.latest_image_b64}"}
                                ]
                            }
                        ]
                        resp_avoid=self.client.responses.create(model=self.model, input=payload)
                        out2_avoid = getattr(resp_avoid, "output_text", None) or ""
                        rospy.loginfo(f"target object satisfied the constraint : {out2_avoid}")
                        if out2_avoid.strip()=="True":
                            avoid_coordinates=self.create_avoid_near_objects(bbox,2.5)
                            rospy.loginfo(f"found {raw}! try to avoid it...")
                            if avoid_coordinates==None:
                                self.avoid_processed = True
                                return
                            self.a_star.add_avoid_points(avoid_coordinates)
                            self.avoid_processed = True
                        else:
                            #해당 물체는 우리가 피해야할 녀석이 아님->gemini 할루시네이션 : partial sg rich버전에서 제거
                            self.false_avoid.add(obj.get("object_id"))
                            rospy.loginfo(f"[task3] '{raw}' not inside box -> remove this object from PSG")
        else:
            objs = (self.last_partial_sg or {}).get("objects", []) or []
            if not objs:
                return

            rospy.loginfo(f"[task3] Processing avoid: {self.avoid}")
            avoid_coordinates = []
            for avoid_object in self.avoid:
                norm = avoid_object.lower().replace(" ", "_")
                for obj in objs:
                    raw = (obj.get("raw_label") or "").lower().replace(" ", "_")
                    if raw == norm:
                        center = obj.get("center", [])
                        if center and len(center) >= 3:
                            avoid_coordinates.append(center)
                            rospy.loginfo(f"[task3] Found avoid object '{avoid_object}' at {center}")
                        break

            if len(avoid_coordinates) == 2:
                blocking_points = self.create_avoid_path_between_objects(
                    avoid_coordinates[0], avoid_coordinates[1], spacing=0.1
                )
                avoid_coordinates.extend(blocking_points)
                self.a_star.add_avoid_points(avoid_coordinates)
                rospy.loginfo(f"[task3] Added {len(avoid_coordinates)} avoid points to A* algorithm")
                self.avoid_processed = True
            return

    def _strip_code_fences(self, text: str) -> dict:
        lines = []
        for ln in (text or "").splitlines():
            s = ln.strip()
            if s.startswith("```") and s.endswith("```"):
                continue
            if s.startswith("```") or s == "```" or s.startswith("```plaintext"):
                continue
            lines.append(ln)
        full_text="\n".join(lines)
        res = {"waypoint": [], "between": [], "avoid": [], "sequence": []}
        for ln in (full_text or "").splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                if s.startswith("waypoint="):
                    res["waypoint"] = json.loads(s[len("waypoint="):])
                elif s.startswith("between="):
                    res["between"]  = json.loads(s[len("between="):])
                elif s.startswith("avoid="):
                    res["avoid"]    = json.loads(s[len("avoid="):])
                elif s.startswith("sequence="):
                    res["sequence"] = json.loads(s[len("sequence="):])
            except Exception as e:
                rospy.logwarn(f"[plan task 3] parse error on line: {s} ({e})")
        return res

    def run(self):
        #1. parse the challenge question
        ##call back challenge qusestion에 주렁주렁 메달아 두기에는 대회에 가면 challenge question은 매초 단위로 발행이 됨-> 수정이 필요한 부분
        while self.challenge_question==None:
            time.sleep(1)
        try:
            rospy.loginfo("[type 3]Asking GPT-4o")
            _prompt_1 = f"""
You are a command-to-plan parser for a mobile robot. 
Given a natural-language instruction, classify it into three directive types.

INPUT QUESTION:
{(self.challenge_question or "").strip()}

OBJECTIVE
- Extract three kinds of directives:
  1) waypoint: go to an object that satisfies constraint.
  2) between: Only when take the path between two objects A and B
  3) avoid: objects to avoid while executing the plan (global constraint)
- Preserve the execution ORDER of actionable steps (waypoint/between) as they appear in the sentence.

NORMALIZATION RULES
- If a constraint exists, set it explicitly.
- If no constraint is implied for a waypoint, use ""(empty string) for constraint.
- If the target object phrase includes a between modifier (e.g., “the vase between the TV and the door”), treat this as a waypoint directive whose constraint is a structured between constraint.

OUTPUT FORMAT (STRICTLY EXACTLY 4 LINES, NO EXTRA TEXT OR EXPLANATION):
waypoint=[{{"id":"way1","object":"<obj>","constraint":<constraint|"">}}, ...]
between=[{{"id":"bet1","a":"<obj_a>","b":"<obj_b>"}}, ...]
avoid=["<obj1>","<obj2>", ...]
sequence=["<way_or_bet_id_in_order>", ...]
- The sequence line MUST reference only waypoint/between IDs (e.g., "way1","bet1","way2") in the intended execution order.
- Do NOT include "avoid" in the sequence
- If a list is empty, output an empty JSON array (e.g., avoid=[], between=[]).
- No code fences, no prose—print exactly those four lines.

EXAMPLES
Q) Go to the bedside table closest to the window and stop at the vase between the TV and the door.
waypoint=[{{"id":"way1","object":"bedside table","constraint":"closest to window"}},{{"id":"way2","object":"vase","constraint":"between TV and the door"}}]
between=[]
avoid=[]
sequence=["way1","way2"]

Q) Walk between the sofa and the table, then go to the door while avoiding the plant.
waypoint=[{{"id":"way1","object":"door","constraint":""}}]
between=[{{"id":"bet1","a":"sofa","b":"table"}}]
avoid=["plant"]
sequence=["bet1","way1"]

NOW OUTPUT FOR THE INPUT QUESTION ABOVE (PRINT ONLY THE FOUR LINES):
"""
            payload=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": _prompt_1}],
                },
            ]
            resp = self.client.responses.create(model=self.model, input=payload)
            out = getattr(resp, "output_text", None) or ""
            rospy.loginfo(f"{out}")
            plan=self._strip_code_fences(out)
            self.waypoint = plan.get("waypoint", []) or []
            self.between  = plan.get("between",  []) or []
            self.avoid    = plan.get("avoid",    []) or []
            self.sequence = plan.get("sequence", []) or []
        except Exception as e:
            err = f"[LLM error] {e}"
            rospy.logerr(err)
            return err
        if len(self.avoid)==0:
            #avoid 없는 버전
            for index in range(len(self.sequence)):
                #새 문제 도전시
                self.visited_list=[]
                self.can_solve=False
                ###문제 해결을 못했다면 여기로 다시 와야합니다!
                while not self.can_solve:
                    self.wait_for_sg=True
                    #partial scene graph 최신화
                    while self.wait_for_sg:
                        self.pub_sg.publish(True)
                        time.sleep(1)
                    self.pub_sg.publish(False)
                    rospy.loginfo(f"self.last_partial_sg={self.last_partial_sg}")
                    rospy.loginfo(f"{self.sequence[index]}")
                    action_1=re.fullmatch(r'(?P<kind>way|bet)(?P<idx>\d+)', self.sequence[index])
                    action=action_1.group('kind')
                    if action=="way":
                        object_info=next((x for x in self.waypoint if x.get('id')==self.sequence[index]),None)
                        _prompt_2 = f"""
System Role:
You are a target selector for a mobile robot. Given a goal constraint/label pair and a Partial Scene Graph (PSG), decide a single movement target.

Input
- Goal sentence: "My current goal is to reach a {{target_label}} that satisfies {{goal_condition}}."
  · goal_condition: "{object_info.get('constraint')}"
  · target_label:   "{object_info.get('object')}"
- Partial Scene Graph (PSG) visible from current pose:
{json.dumps(self.last_partial_sg, ensure_ascii=False)}

- Visited history :
{json.dumps(self.visited_list, ensure_ascii=False)}

PSG Format
- objects: [
    {{"object_id": int, "raw_label": str, "center": [x, y, z]}},
    ...
  ]

Selection Rules (must follow)
1) Candidate Set
   - Normalize common synonyms when obvious: "television"->"tv", "nightstand"->"bedside_table", "couch"->"sofa", "painting"->"picture".
   - Collect all objects whose normalized raw_label equals normalized target_label and normalized object in goal_condition.
   - If a valid candidate is selected, return its center EXACTLY as stored in PSG (no rounding, averaging, or recomputation).
   - If candidate does not exist → return "random_explore".

2) Condition Interpretation (relationship → distance)
   - If multiple candidates exist, select the object that best satisfies the constraint using the "center" coordinates [x, y, z] in PSG.; final tie-breaker: smaller object_id.

3) Visited Constraint (MUST follow)
   - You are given `visited history` as [{{"object_id": int, "center": [x,y,z]}}, ...].
   - Treat an entry in visited history and an object in the PSG as the same object if and only if their object_id values are equal.
   - Do not Return visited object as target_object.

Output (JSON ONLY; no extra text, no code fences). Return exactly one of:

A) Center coordinate:
{{
  "decision": "center",
  "target_object": {{"object_id": <int>, "raw_label": "<str>"}},
  "center": [<float>, <float>, <float>]
}}

B) Random exploration:
{{
  "decision": "random_explore"
}}

Constraints
- Copy the center array verbatim from PSG (preserve original precision).
- Output must be valid JSON only; any other text is forbidden.
"""
                        payload=[
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": _prompt_2}],
                            },
                        ]
                        resp = self.client.responses.create(model=self.model, input=payload)
                        out2 = getattr(resp, "output_text", None) or ""
                        m = re.search(r'\{[\s\S]*\}', out2)
                        data2=json.loads(m.group(0))
                        if data2.get("decision")=="center":
                            obj = self.last_partial_sg_rich['objects']
                            for i in obj:
                                if int(data2.get("target_object").get("object_id"))==int(i["object_id"]):
                                    bbox = i['bbox']
                            x_min=2000;y_min=2000
                            x_max=0;y_max=0
                            for i, (x, y, z) in enumerate(bbox):
                                x,y=self.image_coordinate(x,y,z)
                                if x_min>x:
                                    x_min=x
                                if y_min>y:
                                    y_min=y
                                if x_max<x:
                                    x_max=x
                                if y_max<y:
                                    y_max=y
                            padding=5
                            x_min-=padding;y_min-=padding
                            x_max+=padding;y_max+=padding
                            quad = np.array([
                                [x_min, y_min],
                                [x_max, y_min],
                                [x_max, y_max],
                                [x_min, y_max],
                            ], dtype=np.int32)
                            vis_1=self.latest_cv_img.copy()
                            cv2.polylines(vis_1, [quad], isClosed=True, color=(0,0,255), thickness=2,lineType=cv2.LINE_AA)
                            ok, buf=cv2.imencode(".jpg",vis_1,[int(cv2.IMWRITE_JPEG_QUALITY),90])
                            save_dir = os.path.join(FILE_DIR, "out")
                            os.makedirs(save_dir, exist_ok=True)
                            cv2.imwrite(f"{save_dir}/point.jpg",vis_1)
                            jpeg_bytes=buf.tobytes()
                            self.latest_image_b64=base64.b64encode(jpeg_bytes).decode("ascii")
                            # To do : 이미지로 satisfy 조건 확인
                            _prompt_2_1=f"""According to the image, is the object inside the red box is {object_info.get('object')} {object_info.get('constraint')}? 
                            If there exists any object in the image that better matches the {object_info.get('object')} {object_info.get('constraint')} than the object inside the red box, reply False.
                            Reply with exactly one token: True or False.
                            """
                            payload=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": _prompt_2_1},
                                        {"type": "input_image","image_url": f"data:image/jpg;base64,{self.latest_image_b64}"}
                                    ]
                                }
                            ]
                            resp_1=self.client.responses.create(model=self.model, input=payload)
                            out2_1 = getattr(resp_1, "output_text", None) or ""
                            rospy.loginfo(f"target object satisfied the constraint : {out2_1}")
                            center = data2.get("center")
                            rospy.loginfo(f"[task3] Go to {center}")
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose, center)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
                            #만약 data2.get("satisfied")==True 면 가고 끝 아니면 다시 prompt에 넣고 뻉뻉이
                            if out2_1.strip()=="True":
                                self.can_solve=True
                            elif out2_1.strip()=="False":
                                t_obj=data2.get("target_object",{})
                                obj_id = int(t_obj["object_id"])
                                x,y,z=map(float, center)
                                item={"object_id": obj_id, "center": [x, y, z]}
                                self.visited_list.append(item)
                            else:
                                rospy.logerr(f"What is {data2.get('satisfied')}")
                        elif data2.get("decision")=="random_explore":
                        ##TO_DO--------------------------------------------------------------
                            rospy.loginfo("random_explore!")
                            if self.random_exploration==None:
                                self.random_exploration=random_exploration(self.graph,self.nodes, self.dist_ths)
                            current_goal=self.random_exploration.mst_random_exploration([],self.latest_pose)
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose,current_goal)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
                        ##-------------------------------------------------------------------
                    elif action=="bet":
                        object_info_2=next((x for x in self.between if x.get('id')==self.sequence[index]),None)
                        _prompt_3=f"""
                    System Role:
You are a selector for a mobile robot. You will receive TWO target object labels and a Partial Scene Graph (PSG). Decide what to return based on how many of the two object *types* exist in the PSG.

Inputs
- target_a: "{object_info_2.get("a")}"
- target_b: "{object_info_2.get("b")}"
- Partial Scene Graph (PSG):
{json.dumps(self.last_partial_sg, ensure_ascii=False)}
- Visited history from current episode:
{json.dumps(self.visited_list, ensure_ascii=False)}

PSG Format
- objects: [
    {{"object_id": int, "raw_label": str, "center": [x, y, z]}},
    ...
  ]
- relationships: {{ ... }}   # may exist; not required for this task

Normalization & Matching (must follow)
1) Normalize labels to compare:
   - lowercase
   - spaces -> underscores (e.g., "tv cabinet" -> "tv_cabinet")
   - obvious synonyms: "television"->"tv", "nightstand"->"bedside_table", "couch"->"sofa"
2) A PSG object is considered a match if normalized(raw_label) == normalized(target_label).

Multiplicity Rules
- If multiple instances exist for a given type:
  • For the TWO-TYPES case: choose the pair (a_i, b_j) that minimizes Euclidean distance between centers. Tie-brealers : smaller euclidean distance between center
  • For the ONE-TYPE case: choose the instance with the smallest object_id.

Decision Logic (exactly one applies)
A) BOTH types exist (at least one instance of each):
   → Return the midpoint (arithmetic mean) of the chosen pair’s centers:
      center = [ (ax+bx)/2, (ay+by)/2, (az+bz)/2 ]
B) EXACTLY ONE type exists:
    - You are given `visited history` as [{{"object_id": int, "center": [x,y,z]}}, ...].
    Consider ONLY instances of the existing type whose object_id is NOT present in the Visited history.
    - If at least one unvisited instance exists, choose deterministically:
         1) Prefer the instance with the smallest object_id.
   → Return the chosen instance’s center (copied from PSG).
   → If ALL instances of the existing type are already visited, RETURN random_explore
C) NEITHER type exists:
   → Return random_explore.

Output (JSON ONLY; no extra text, no code fences)
- Case A (both types exist):
{{
  "decision": "between",
  "a": {{"object_id": <int>, "raw_label": "<str>", "center": [<float>, <float>, <float>]}},
  "b": {{"object_id": <int>, "raw_label": "<str>", "center": [<float>, <float>, <float>]}},
  "center": [<float>, <float>, <float>]
}}
- Case B (exactly one type exists):
{{
  "decision": "center",
  "target_object": {{"object_id": <int>, "raw_label": "<str>", "center": [<float>, <float>, <float>]}},
  "center": [<float>, <float>, <float>]
}}
- Case C (neither exists):
{{
  "decision": "random_explore"
}}

Constraints
- Do NOT fabricate centers. Copy centers from PSG for the chosen objects (then compute the midpoint only for the two-types case).
- Preserve numeric precision (no rounding). Output must be valid JSON only.
"""
                        #Todo : between 문제 풀기
                        payload=[
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": _prompt_3}],
                            },
                        ]
                        resp = self.client.responses.create(model=self.model, input=payload)
                        out3=getattr(resp, "output_text", None) or ""
                        rospy.loginfo(f"{out3}")
                        m=re.search(r'\{[\s\S]*\}',out3)
                        data3=json.loads(m.group(0))
                        if data3.get("decision")=="between":
                            center = data3.get("center")
                            #Todo : center 문제풀기
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose, center)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
                            rospy.loginfo(f"[task3] Go to {center}")
                            self.can_solve=True
                        elif data3.get("decision")=="center":
                            center = data3.get("center")
                            #해당 좌표로 이동
                            rate = rospy.Rate(5)
                            while not rospy.is_shutdown():
                                move_done = self.a_star.move(self.latest_pose, center)
                                if move_done:
                                    break
                                rate.sleep()
                            rospy.loginfo(f"[task3] Go to {center}")
                            self.can_solve=True
                            t_obj=data3.get("target_object",{})
                            obj_id = int(t_obj["object_id"])
                            x,y,z=map(float, center)
                            item={"object_id": obj_id, "center": [x, y, z]}
                            self.visited_list.append(item)
                        elif data3.get("decision")=="random_explore":
                        #between 이면 가고 끝 center면 가고 다시 prompt  random explore면 탐색
                            rospy.loginfo("random_explore!")
                            if self.random_exploration==None:
                                self.random_exploration=random_exploration(self.graph,self.nodes, self.dist_ths)
                            current_goal=self.random_exploration.mst_random_exploration([],self.latest_pose)
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose, current_goal)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
        else:
            for index in range(len(self.sequence)):
                #새 문제 도전시
                self.visited_list=[]
                self.can_solve=False
                ###문제 해결을 못했다면 여기로 다시 와야합니다!
                while not self.can_solve:
                    self.wait_for_sg=True
                    #partial scene graph 최신화
                    while self.wait_for_sg:
                        self.pub_sg.publish(True)
                        time.sleep(1)
                    self.pub_sg.publish(False)
                    rospy.loginfo(f"self.last_partial_sg={self.last_partial_sg}")
                    self._maybe_process_avoid_once()
                    rospy.loginfo(f"{self.sequence[index]}")
                    action_1=re.fullmatch(r'(?P<kind>way|bet)(?P<idx>\d+)', self.sequence[index])
                    action=action_1.group('kind')
                    if action=="way":
                        object_info=next((x for x in self.waypoint if x.get('id')==self.sequence[index]),None)
                        _prompt_2 = f"""
System Role:
You are a target selector for a mobile robot. Given a goal constraint/label pair and a Partial Scene Graph (PSG), decide a single movement target.

Input
- Goal sentence: "My current goal is to reach a {{target_label}} that satisfies {{goal_condition}}."
  · goal_condition: "{object_info.get('constraint')}"
  · target_label:   "{object_info.get('object')}"
- Partial Scene Graph (PSG) visible from current pose:
{json.dumps(self.last_partial_sg, ensure_ascii=False)}

- Visited history :
{json.dumps(self.visited_list, ensure_ascii=False)}

PSG Format
- objects: [
    {{"object_id": int, "raw_label": str, "center": [x, y, z]}},
    ...
  ]

Selection Rules (must follow)
1) Candidate Set
   - Normalize common synonyms when obvious: "television"->"tv", "nightstand"->"bedside_table", "couch"->"sofa", "painting"->"picture".
   - Collect all objects whose normalized raw_label equals normalized target_label and normalized object in goal_condition.
   - If a valid candidate is selected, return its center EXACTLY as stored in PSG (no rounding, averaging, or recomputation).
   - If candidate does not exist → return "random_explore".

2) Condition Interpretation (relationship → distance)
   - If multiple candidates exist, select the object that best satisfies the constraint using the "center" coordinates [x, y, z] in PSG.; final tie-breaker: smaller object_id.

3) Visited Constraint (MUST follow)
   - You are given `visited history` as [{{"object_id": int, "center": [x,y,z]}}, ...].
   - Treat an entry in visited history and an object in the PSG as the same object if and only if their object_id values are equal.
   - Do not Return visited object as target_object.

Output (JSON ONLY; no extra text, no code fences). Return exactly one of:

A) Center coordinate:
{{
  "decision": "center",
  "target_object": {{"object_id": <int>, "raw_label": "<str>"}},
  "center": [<float>, <float>, <float>]
}}

B) Random exploration:
{{
  "decision": "random_explore"
}}

Constraints
- Copy the center array verbatim from PSG (preserve original precision).
- Output must be valid JSON only; any other text is forbidden.
"""
                        payload=[
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": _prompt_2}],
                            },
                        ]
                        resp = self.client.responses.create(model=self.model, input=payload)
                        out2 = getattr(resp, "output_text", None) or ""
                        m = re.search(r'\{[\s\S]*\}', out2)
                        data2=json.loads(m.group(0))
                        if data2.get("decision")=="center":
                            obj = self.last_partial_sg_rich['objects']
                            for i in obj:
                                if int(data2.get("target_object").get("object_id"))==int(i["object_id"]):
                                    bbox = i['bbox']
                            x_min=2000;y_min=2000
                            x_max=0;y_max=0
                            for i, (x, y, z) in enumerate(bbox):
                                x,y=self.image_coordinate(x,y,z)
                                if x_min>x:
                                    x_min=x
                                if y_min>y:
                                    y_min=y
                                if x_max<x:
                                    x_max=x
                                if y_max<y:
                                    y_max=y
                            padding=5
                            x_min-=padding;y_min-=padding
                            x_max+=padding;y_max+=padding
                            quad = np.array([
                                [x_min, y_min],
                                [x_max, y_min],
                                [x_max, y_max],
                                [x_min, y_max],
                            ], dtype=np.int32)
                            vis_2=self.latest_cv_img.copy()
                            cv2.polylines(vis_2, [quad], isClosed=True, color=(0,0,255), thickness=2,lineType=cv2.LINE_AA)
                            ok, buf=cv2.imencode(".jpg",vis_2,[int(cv2.IMWRITE_JPEG_QUALITY),90])
                            save_dir = os.path.join(FILE_DIR, "out")
                            os.makedirs(save_dir, exist_ok=True)
                            cv2.imwrite(f"{save_dir}/point.jpg",vis_2)
                            jpeg_bytes=buf.tobytes()
                            self.latest_image_b64=base64.b64encode(jpeg_bytes).decode("ascii")
                            # To do : 이미지로 satisfy 조건 확인
                            _prompt_2_1=f"""According to the image, is the object inside the red box is {object_info.get('object')} {object_info.get('constraint')}? 
                            If there exists any object in the image that better matches the {object_info.get('object')} {object_info.get('constraint')} than the object inside the red box, reply False.
                            Reply with exactly one token: True or False.
                            """
                            payload=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": _prompt_2_1},
                                        {"type": "input_image","image_url": f"data:image/jpg;base64,{self.latest_image_b64}"}
                                    ]
                                }
                            ]
                            resp_1=self.client.responses.create(model=self.model, input=payload)
                            out2_1 = getattr(resp_1, "output_text", None) or ""
                            rospy.loginfo(f"{out2_1}")
                            center = data2.get("center")
                            rospy.loginfo(f"[task3] Go to {center}")
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose, center)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
                            #만약 data2.get("satisfied")==True 면 가고 끝 아니면 다시 prompt에 넣고 뻉뻉이
                            if out2_1.strip()=="True":
                                self.can_solve=True
                            elif out2_1.strip()=="False":
                                t_obj=data2.get("target_object",{})
                                obj_id = int(t_obj["object_id"])
                                x,y,z=map(float, center)
                                item={"object_id": obj_id, "center": [x, y, z]}
                                self.visited_list.append(item)
                            else:
                                rospy.logerr(f"What is {data2.get('satisfied')}")
                        elif data2.get("decision")=="random_explore":
                        ##TO_DO--------------------------------------------------------------
                            rospy.loginfo("random_explore!")
                            if self.random_exploration==None:
                                self.random_exploration=random_exploration(self.graph,self.nodes, self.dist_ths)
                            current_goal=self.random_exploration.mst_random_exploration([],self.latest_pose)
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose,current_goal)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
                        ##-------------------------------------------------------------------
                    elif action=="bet":
                        object_info_2=next((x for x in self.between if x.get('id')==self.sequence[index]),None)
                        _prompt_3=f"""
                    System Role:
You are a selector for a mobile robot. You will receive TWO target object labels and a Partial Scene Graph (PSG). Decide what to return based on how many of the two object *types* exist in the PSG.

Inputs
- target_a: "{object_info_2.get("a")}"
- target_b: "{object_info_2.get("b")}"
- Partial Scene Graph (PSG):
{json.dumps(self.last_partial_sg, ensure_ascii=False)}
- Visited history from current episode:
{json.dumps(self.visited_list, ensure_ascii=False)}

PSG Format
- objects: [
    {{"object_id": int, "raw_label": str, "center": [x, y, z]}},
    ...
  ]
- relationships: {{ ... }}   # may exist; not required for this task

Normalization & Matching (must follow)
1) Normalize labels to compare:
   - lowercase
   - spaces -> underscores (e.g., "tv cabinet" -> "tv_cabinet")
   - obvious synonyms: "television"->"tv", "nightstand"->"bedside_table", "couch"->"sofa"
2) A PSG object is considered a match if normalized(raw_label) == normalized(target_label).

Multiplicity Rules
- If multiple instances exist for a given type:
  • For the TWO-TYPES case: choose the pair (a_i, b_j) that minimizes Euclidean distance between centers. Tie-brealers : smaller euclidean distance between center
  • For the ONE-TYPE case: choose the instance with the smallest object_id.

Decision Logic (exactly one applies)
A) BOTH types exist (at least one instance of each):
   → Return the midpoint (arithmetic mean) of the chosen pair’s centers:
      center = [ (ax+bx)/2, (ay+by)/2, (az+bz)/2 ]
B) EXACTLY ONE type exists:
    - You are given `visited history` as [{{"object_id": int, "center": [x,y,z]}}, ...].
    Consider ONLY instances of the existing type whose object_id is NOT present in the Visited history.
    - If at least one unvisited instance exists, choose deterministically:
         1) Prefer the instance with the smallest object_id.
   → Return the chosen instance’s center (copied from PSG).
   → If ALL instances of the existing type are already visited, RETURN random_explore
C) NEITHER type exists:
   → Return random_explore.

Output (JSON ONLY; no extra text, no code fences)
- Case A (both types exist):
{{
  "decision": "between",
  "a": {{"object_id": <int>, "raw_label": "<str>", "center": [<float>, <float>, <float>]}},
  "b": {{"object_id": <int>, "raw_label": "<str>", "center": [<float>, <float>, <float>]}},
  "center": [<float>, <float>, <float>]
}}
- Case B (exactly one type exists):
{{
  "decision": "center",
  "target_object": {{"object_id": <int>, "raw_label": "<str>", "center": [<float>, <float>, <float>]}},
  "center": [<float>, <float>, <float>]
}}
- Case C (neither exists):
{{
  "decision": "random_explore"
}}

Constraints
- Do NOT fabricate centers. Copy centers from PSG for the chosen objects (then compute the midpoint only for the two-types case).
- Preserve numeric precision (no rounding). Output must be valid JSON only.
"""
                        #Todo : between 문제 풀기
                        payload=[
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": _prompt_3}],
                            },
                        ]
                        resp = self.client.responses.create(model=self.model, input=payload)
                        out3=getattr(resp, "output_text", None) or ""
                        rospy.loginfo(f"{out3}")
                        m=re.search(r'\{[\s\S]*\}',out3)
                        data3=json.loads(m.group(0))
                        if data3.get("decision")=="between":
                            center = data3.get("center")
                            #Todo : center 문제풀기
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose, center)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
                            rospy.loginfo(f"[task3] Go to {center}")
                            self.can_solve=True
                        elif data3.get("decision")=="center":
                            center = data3.get("center")
                            #해당 좌표로 이동
                            rate = rospy.Rate(5)
                            while not rospy.is_shutdown():
                                move_done = self.a_star.move(self.latest_pose, center)
                                if move_done:
                                    break
                                rate.sleep()
                            rospy.loginfo(f"[task3] Go to {center}")
                            self.can_solve=True
                            t_obj=data3.get("target_object",{})
                            obj_id = int(t_obj["object_id"])
                            x,y,z=map(float, center)
                            item={"object_id": obj_id, "center": [x, y, z]}
                            self.visited_list.append(item)
                        elif data3.get("decision")=="random_explore":
                        #between 이면 가고 끝 center면 가고 다시 prompt  random explore면 탐색
                            rospy.loginfo("random_explore!")
                            if self.random_exploration==None:
                                self.random_exploration=random_exploration(self.graph,self.nodes, self.dist_ths)
                            current_goal=self.random_exploration.mst_random_exploration([],self.latest_pose)
                            rate = rospy.Rate(5)
                            # self.a_star.move가 True를 반환하거나, ROS 노드가 종료될 때까지 반복합니다.
                            while not rospy.is_shutdown():
                                # move 함수를 호출하고 결과를 변수에 저장합니다.
                                move_done = self.a_star.move(self.latest_pose, current_goal)
                                # 만약 성공했다면(True 반환), 루프를 빠져나갑니다.
                                if move_done:
                                    break
                                # 아직 성공하지 못했다면(False 반환), 다음 시도까지 0.2초 대기합니다.
                                rate.sleep()
        self.shutdown.publish(True)

def main():
    rospy.init_node("plan_task_3", anonymous=False)
    node = plan_task_3()
    node.run()
    rospy.spin()

if __name__ == "__main__":
    main()
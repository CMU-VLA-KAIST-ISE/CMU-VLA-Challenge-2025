import sys, os, json, math, re
import rospy
import numpy as np

from collections import defaultdict, deque

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
        
        # 이전 avoid_list와 다른지 확인
        avoid_go = False
        if self.latest_avoid_list != avoid_list :
            avoid_go = True
        
        self.latest_avoid_list = avoid_list

        # 1) 회피 노드 제거한 인접 리스트 구성 및 최인접 노드 반환
        pruned = defaultdict(list)
        except_node_list = set(self.visited_mst_list)

        if avoid_go:
            nodes_to_avoid = {exp_node for exp_node in self.graph.keys() if np.linalg.norm(np.array([p[:2] for p in avoid_list]) - np.array(self.nodes[exp_node][:2]), axis=1).min() < self.dist_ths}
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
        
        # naive option) 가장 가까운 노드를 선택하되, 현재 위치와 충분히 떨어진 노드를 찾습니다. (밑의 코드 에러 많이 발생하면 이것으로 대체)
        # goal_node_idx = None
        # for n_idx in sorted_candidates:
        #     dist_to_node = np.linalg.linalg.norm(np.array(self.nodes[n_idx][:2]) - self.latest_pose[:2])
        #     if dist_to_node > self.dist_ths: 
        #         goal_node_idx = n_idx
        #         break
        
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
        #rospy.loginfo(f"[DEBUG] nearest_node: {nearest_node}, nearest_ths_node: {nearest_ths_node}, dist_ths: {self.dist_ths}")
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
    

if __name__ == "__main__":
    """
    HOW TO USE
    random_exploration_ = random_exploration(graph, nodes, dist_ths)
    current_goal = random_exploration_.mst_random_exploration(avoid_list, latest_pose)
    """
    pass
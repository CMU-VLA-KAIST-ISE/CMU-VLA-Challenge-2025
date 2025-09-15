#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
import numpy as np
from typing import List, Tuple, Optional


class AStarPointNavigator:
    def __init__(self):
        rospy.init_node('a_star_point_navigator')

        # Params
        self.frame = rospy.get_param('~goal_frame', 'map')
        self.stand_off = float(rospy.get_param('~stand_off', 0.3))
        self.a_star_node_size = float(rospy.get_param('~a_star_node_size', 0.5))
        self.reach_thresh = float(rospy.get_param('~reach_thresh', 0.4))
        self.wp_rate = float(rospy.get_param('~waypoint_rate', 10.0))
        self.max_segment = float(rospy.get_param('~max_segment', 0.6))
        self.min_goal_separation = float(rospy.get_param('~min_goal_separation', 0.3))

        # Buffers
        self.a_nodes: Optional[List[Tuple[float, float, float]]] = None
        self.current_pose: Optional[np.ndarray] = None
        self.goal_center: Optional[np.ndarray] = None
        self.route: List[Tuple[float, float, float]] = []
        self.route_idx: int = 0

        # Publishers
        self.pub_wp = rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=10)
        self.pub_status = rospy.Publisher('/a_star_point_nav/status', String, queue_size=10)

        # Subscribers
        rospy.Subscriber('/a_node_list', PointCloud2, self.cb_a_nodes, queue_size=1)
        rospy.Subscriber('/state_estimation', Odometry, self.cb_odom, queue_size=20)
        # Only one goal topic: Pose2D center in goal_frame
        rospy.Subscriber('/semantic_goal_center', Pose2D, self.cb_goal_p2d, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.wp_rate, 1.0)), self.on_timer)
        rospy.loginfo('[AStarPointNavigator] ready. stand_off=%.2f reach=%.2f rate=%.1f', self.stand_off, self.reach_thresh, self.wp_rate)

    # ---------------- Callbacks ----------------
    def cb_a_nodes(self, msg: PointCloud2):
        self.a_nodes = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.current_pose = np.array([p.x, p.y, p.z], dtype=float)

    def cb_goal_p2d(self, msg: Pose2D):
        # Pose2D carries x,y,theta in the configured goal frame
        self.goal_center = np.array([msg.x, msg.y, 0.0], dtype=float)
        self.plan_route()

    # ---------------- Planning ----------------
    def _nearest_idx(self, pts: List[Tuple[float, float, float]], target_xy: np.ndarray) -> Optional[int]:
        if not pts:
            return None
        arr = np.asarray(pts, dtype=float)
        d = np.linalg.norm(arr[:, :2] - target_xy[:2], axis=1)
        return int(np.argmin(d))

    def _neighbors(self, pts: List[Tuple[float, float, float]], idx: int, threshold: float) -> List[int]:
        if not pts:
            return []
        origin = np.asarray(pts[idx], dtype=float)
        arr = np.asarray(pts, dtype=float)
        diff = np.abs(arr[:, :2] - origin[:2])
        mask = (np.maximum(diff[:, 0], diff[:, 1]) <= threshold)
        nbrs = np.where(mask)[0].tolist()
        if idx in nbrs:
            nbrs.remove(idx)
        return nbrs

    def _heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _a_star(self, pts: List[Tuple[float, float, float]], start_idx: int, goal_idx: int, threshold: float) -> List[int]:
        if start_idx is None or goal_idx is None:
            return []
        if start_idx == goal_idx:
            return [start_idx]
        open_set = {start_idx}
        came_from = {}
        g = {start_idx: 0.0}
        arr = np.asarray(pts, dtype=float)
        f = {start_idx: self._heuristic(arr[start_idx], arr[goal_idx])}
        visited = set()
        while open_set:
            current = min(open_set, key=lambda x: f.get(x, 1e18))
            if current == goal_idx:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            open_set.remove(current)
            visited.add(current)
            for nb in self._neighbors(pts, current, threshold):
                if nb in visited:
                    continue
                tentative_g = g[current] + self._heuristic(arr[current], arr[nb])
                if tentative_g < g.get(nb, 1e18):
                    came_from[nb] = current
                    g[nb] = tentative_g
                    f[nb] = tentative_g + self._heuristic(arr[nb], arr[goal_idx])
                    open_set.add(nb)
        return []

    def _choose_goal_node(self, center: np.ndarray) -> Optional[int]:
        if not self.a_nodes:
            return None
        # compute standoff point from center away from robot
        if self.current_pose is None:
            rospy.logwarn_throttle(1.0, '[AStarPointNavigator] no current pose; using raw center as goal')
            target_xy = center[:2]
        else:
            dir_vec = self.current_pose[:2] - center[:2]
            n = np.linalg.norm(dir_vec)
            if n < 1e-6:
                target_xy = center[:2]
            else:
                target_xy = center[:2] + (self.stand_off * dir_vec / n)
        # snap to nearest a* node
        return self._nearest_idx(self.a_nodes, np.array([target_xy[0], target_xy[1], 0.0], dtype=float))

    def plan_route(self):
        if self.goal_center is None or self.a_nodes is None or self.current_pose is None:
            rospy.logwarn_throttle(1.0, '[AStarPointNavigator] plan skipped (goal/a_nodes/pose missing)')
            return
        start_idx = self._nearest_idx(self.a_nodes, self.current_pose)
        goal_idx = self._choose_goal_node(self.goal_center)
        if start_idx is None or goal_idx is None:
            rospy.logwarn('[AStarPointNavigator] failed to pick start/goal node')
            return
        # If the grid is sparse and start == goal, pick an alternate goal node at least min_goal_separation away
        if start_idx == goal_idx:
            arr = np.asarray(self.a_nodes, dtype=float)
            pose_xy = self.current_pose[:2]
            # distance from current pose
            d_from_pose = np.linalg.norm(arr[:, :2] - pose_xy[None, :], axis=1)
            # distance to desired target point (goal_center)
            d_to_target = np.linalg.norm(arr[:, :2] - self.goal_center[:2][None, :], axis=1)
            candidates = np.where(d_from_pose >= self.min_goal_separation)[0]
            if candidates.size > 0:
                # choose the one closest to target among candidates
                goal_idx = int(candidates[np.argmin(d_to_target[candidates])])
                rospy.loginfo('[AStarPointNavigator] adjusted goal_idx to %d due to sparse grid (min_sep=%.2f)', goal_idx, self.min_goal_separation)
            else:
                # as a last resort, synthesize a short step towards goal
                dir_vec = self.goal_center[:2] - pose_xy
                n = np.linalg.norm(dir_vec)
                if n > 1e-6:
                    step = min(self.max_segment, max(self.reach_thresh, 0.2))
                    target_xy = pose_xy + (dir_vec / n) * step
                    self.route = [(float(target_xy[0]), float(target_xy[1]), float(self.current_pose[2]))]
                    self.route_idx = 0
                    rospy.logwarn('[AStarPointNavigator] synthesized 1-step route to avoid stiction')
                    return
        threshold = self.a_star_node_size + 0.05
        idx_path = self._a_star(self.a_nodes, start_idx, goal_idx, threshold)
        if not idx_path:
            rospy.logwarn('[AStarPointNavigator] A* failed: no path')
            self.route = []
            self.route_idx = 0
            return
        self.route = [self.a_nodes[i] for i in idx_path]
        self.route_idx = 0
        # rospy.loginfo('[AStarPointNavigator] planned route with %d waypoints', len(self.route))

    # ---------------- Control ----------------
    def on_timer(self, event):
        if self.current_pose is None or not self.route:
            return
        # advance along route
        if self.route_idx >= len(self.route):
            return
        cur_wp = np.array(self.route[self.route_idx], dtype=float)
        cur_xy = self.current_pose[:2]
        d = np.linalg.norm(cur_wp[:2] - cur_xy)
        rospy.loginfo_throttle(0.5, '[AStarNav] wp=(%.3f,%.3f) cur=(%.3f,%.3f) d=%.3f idx=%d/%d',
                               cur_wp[0], cur_wp[1], cur_xy[0], cur_xy[1], d, self.route_idx, len(self.route))
        if d < self.reach_thresh:
            self.route_idx += 1
            if self.route_idx >= len(self.route):
                self.pub_status.publish(String('arrived'))
                return
            cur_wp = np.array(self.route[self.route_idx], dtype=float)
        # cap segment length to avoid oscillation
        vec = cur_wp[:2] - cur_xy
        dist = np.linalg.norm(vec)
        if dist > self.max_segment:
            vec = vec * (self.max_segment / max(dist, 1e-9))
        target_xy = cur_xy + vec
        yaw = float(np.arctan2(vec[1], vec[0])) if np.linalg.norm(vec) > 1e-6 else 0.0
        cmd = Pose2D()
        cmd.x = float(target_xy[0])
        cmd.y = float(target_xy[1])
        cmd.theta = yaw
        self.pub_wp.publish(cmd)


def main():
    try:
        AStarPointNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

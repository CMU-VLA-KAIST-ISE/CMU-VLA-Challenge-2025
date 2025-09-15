# -*- coding: utf-8 -*-
"""
Lightweight Scene Graph Builder for online ROS fusion pipeline.

Inputs (per object in MAP frame):
- id: int
- label: str
- center: [x,y,z]
- aabb: {"min":[x,y,z], "max":[x,y,z]}  # axis-aligned bbox in MAP
- footprint: [xmin, ymin, xmax, ymax]   # top-down XY rectangle
- npoints: int

Outputs:
scene_graph = {
  "frame": str,
  "timestamp": float,
  "objects": [{...}],
  "relations": {
     "near": {str(id): [id,...]}, "above": {...}, "below": {...},
     "on": {...}, "left_of": {...}, "right_of": {...},
     "in_front_of": {...}, "behind": {...}, "between": {...}
  }
}
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import math
import numpy as np

@dataclass
class SceneGraphParams:
    near_dist: float = 0.8                # m
    overlap_ratio: float = 0.2            # horizontal overlap ratio (intersection / min(area))
    on_gap: float = 0.15                  # m (vertical gap for 'on')
    above_gap: float = 0.20               # m (min z gap for 'above/below')
    lr_thresh: float = 0.20               # m lateral threshold (y-axis in robot frame)
    fb_thresh: float = 0.20               # m forward/back threshold (x-axis in robot frame)
    between_lat_thresh: float = 0.30      # m lateral distance to BC line in XY
    max_between_pairs_per_anchor: int = 2 # avoid explosion

def _rect_intersection_area(a, b):
    # a,b = [xmin, ymin, xmax, ymax]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)

def _rect_area(r):
    return max(0.0, (r[2] - r[0])) * max(0.0, (r[3] - r[1]))

def _horizontal_overlap_ratio(a, b):
    inter = _rect_intersection_area(a, b)
    ma = _rect_area(a); mb = _rect_area(b)
    denom = min(ma, mb) if min(ma, mb) > 1e-9 else (ma + mb + 1e-9)
    return inter / denom

def _rotate_into_robot_frame(xy: np.ndarray, yaw: float) -> np.ndarray:
    # map -> robot frame (robot x forward, y left)
    c = math.cos(-yaw); s = math.sin(-yaw)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    return (R @ xy.T).T

def build_scene_graph(
    objects: List[Dict[str, Any]],
    frame: str,
    robot_xyyaw: List[float],  # [rx, ry, ryaw] in MAP
    params: SceneGraphParams = SceneGraphParams()
) -> Dict[str, Any]:
    G = {
        "frame": frame,
        "timestamp": float(np.nan),  # caller sets if needed
        "objects": [],
        "relations": {k:{} for k in ["near","above","below","on","left_of","right_of","in_front_of","behind","between"]}
    }

    if not objects:
        return G

    # pack arrays for vector ops
    ids   = np.array([o["id"] for o in objects], dtype=int)
    ctrs  = np.array([o["center"] for o in objects], dtype=np.float32)       # (N,3)
    aabbs = np.array([[*o["aabb"]["min"], *o["aabb"]["max"]] for o in objects], dtype=np.float32)  # (N,6)
    fp    = np.array([o["footprint"] for o in objects], dtype=np.float32)    # (N,4) [xmin,ymin,xmax,ymax]

    for o in objects:
        G["objects"].append({
            "id": int(o["id"]),
            "label": o.get("label",""),
            "center": [float(v) for v in o["center"]],
            "aabb": {"min":[float(v) for v in o["aabb"]["min"]],
                     "max":[float(v) for v in o["aabb"]["max"]]},
            "footprint": [float(v) for v in o["footprint"]],
            "npoints": int(o.get("npoints",0)),
        })

    N = len(objects)
    # precompute robot-frame XY for LR/FB relations
    rx, ry, ryaw = robot_xyyaw
    xy_map = ctrs[:, :2]
    xy_robot = _rotate_into_robot_frame(xy_map - np.array([[rx,ry]], dtype=np.float32), ryaw)

    # pairwise relations
    for i in range(N):
        aid = str(int(ids[i]))
        G["relations"]["near"][aid] = []
        G["relations"]["above"][aid] = []
        G["relations"]["below"][aid] = []
        G["relations"]["on"][aid] = []
        G["relations"]["left_of"][aid] = []
        G["relations"]["right_of"][aid] = []
        G["relations"]["in_front_of"][aid] = []
        G["relations"]["behind"][aid] = []
        G["relations"]["between"][aid] = []

        for j in range(N):
            if i == j: continue
            # NEAR
            d = np.linalg.norm(ctrs[i] - ctrs[j])
            if d < params.near_dist:
                G["relations"]["near"][aid].append(int(ids[j]))

            # horizontal overlap for vertical relations
            r = _horizontal_overlap_ratio(fp[i], fp[j])

            # ABOVE / BELOW (z gap + sufficient horizontal overlap)
            dz = ctrs[j][2] - ctrs[i][2]  # j relative to i
            if (dz > params.above_gap) and (r > params.overlap_ratio):
                G["relations"]["above"][aid].append(int(ids[j]))
            if (dz < -params.above_gap) and (r > params.overlap_ratio):
                G["relations"]["below"][aid].append(int(ids[j]))

            # ON (j on i): j is above i, small vertical gap, high horizontal overlap
            top_i = aabbs[i, 5]  # max z of i
            bot_j = aabbs[j, 2]  # min z of j
            gap = bot_j - top_i  # >= 0
            if (gap >= -0.02) and (gap <= params.on_gap) and (r > max(0.5, params.overlap_ratio)):
                G["relations"]["on"][aid].append(int(ids[j]))

            # LEFT/RIGHT & FRONT/BEHIND (robot frame)
            rel = xy_robot[j] - xy_robot[i]  # in robot frame
            if rel[1] > params.lr_thresh:
                G["relations"]["left_of"][aid].append(int(ids[j]))
            elif rel[1] < -params.lr_thresh:
                G["relations"]["right_of"][aid].append(int(ids[j]))
            if rel[0] > params.fb_thresh:
                G["relations"]["in_front_of"][aid].append(int(ids[j]))
            elif rel[0] < -params.fb_thresh:
                G["relations"]["behind"][aid].append(int(ids[j]))

        # BETWEEN (A between B and C) — cheap top-down test
        # pick up to K best pairs by lateral distance
        best = []
        for b in range(N):
            if b == i: continue
            for c in range(b+1, N):
                if c == i: continue
                B = ctrs[b][:2]; C = ctrs[c][:2]; A = ctrs[i][:2]
                BC = C - B; L2 = float(BC @ BC)
                if L2 < 1e-6: continue
                t = float((A - B) @ BC) / L2
                if t <= 0.0 or t >= 1.0: continue
                P = B + t * BC
                lat = float(np.linalg.norm(A - P))
                best.append((lat, int(ids[b]), int(ids[c])))
        best.sort(key=lambda x: x[0])
        for lat, bb, cc in best[:params.max_between_pairs_per_anchor]:
            if lat < params.between_lat_thresh:
                G["relations"]["between"][aid].append([bb, cc])

    return G

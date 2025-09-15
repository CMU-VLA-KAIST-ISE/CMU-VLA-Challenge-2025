#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import rospy
import json
import numpy as np
from datetime import datetime
import math
from PIL import Image as PILImage
# ROS 메시지 타입
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import Image
from std_msgs.msg import String

# TF2 및 변환 라이브러리
import tf2_ros
import tf.transformations as tft
from cv_bridge import CvBridge, CvBridgeError
import cv2



# 3. Occlusion 개선: 구조물 및 얇은 객체 정의
STRUCTURE_LABELS = {"wall", "floor", "ceiling", "window", "door", "pillar", "rail", "air vent", "airvent", "stairs"}
THICKNESS_EPSILON = 0.03  # 3cm 미만 두께의 객체는 가림 계산에서 제외

class PartialSceneGraphGenerator:
    def __init__(self):
        rospy.init_node('partial_scene_graph_generator', anonymous=False)

        # ----- 설정 변수 -----
        self.scene_name = "livingroom_1"
        self.camera_fov_deg = 120
        self.generation_interval_sec = 5.0

        # ----- 경로 설정 -----
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.abspath(os.path.join(script_dir, "../../../../"))
        self.scene_files_path = os.path.join(self.base_path, "system/unity/src/vehicle_simulator/mesh/unity")
        self.save_dir_sg = os.path.join(self.base_path, "collected_data", self.scene_name, "partials")
        self.save_dir_img = os.path.join(self.base_path, "collected_data", self.scene_name, "images")
        os.makedirs(self.save_dir_sg, exist_ok=True)
        os.makedirs(self.save_dir_img, exist_ok=True)

        # ----- 데이터 로드 -----
        self.full_scene_graph = self._load_full_scene_graph()
        self.all_objects = self._extract_all_objects()
        if not self.full_scene_graph:
            rospy.logerr("Failed to load scene graph. Shutting down.")
            rospy.signal_shutdown("Data loading failed")
            return

        # ----- 상태 변수 및 TF 리스너 -----
        self.latest_image_msg = None
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_frame = None
        self.camera_intrinsics = None  # K 행렬 (fx, fy, cx, cy, w, h)

        # ----- ROS Subscriber & Publisher -----
        rospy.Subscriber("/camera/image", Image, self.image_callback, queue_size=1)
        self.pose_pub = rospy.Publisher("~current_pose", PoseStamped, queue_size=1)
        self.image_pub = rospy.Publisher("~current_image", Image, queue_size=1)
        self.partial_sg_pub = rospy.Publisher("~partial_scene_graph", String, queue_size=1)

        rospy.loginfo("Partial Scene Graph Generator initialized with TF2 and advanced occlusion logic.")
        rospy.loginfo(f"Data will be saved in: {self.save_dir_sg} and {self.save_dir_img}")

    def run(self):
        rospy.loginfo("Waiting for the first image message to get camera info...")
        while not rospy.is_shutdown() and self.camera_intrinsics is None:
            rospy.sleep(0.5)
        rospy.loginfo("Camera info received. Starting generation loop.")
        
        rospy.Timer(rospy.Duration(self.generation_interval_sec), self.generate_and_publish)
        rospy.spin()

    def image_callback(self, msg):
        self.latest_image_msg = msg
        if self.camera_frame is None:
            self.camera_frame = msg.header.frame_id or "camera_color_optical_frame"
        if self.camera_intrinsics is None:
            w, h = msg.width, msg.height
            hfov = np.deg2rad(self.camera_fov_deg)
            fx = w / (2.0 * np.tan(hfov / 2.0))
            fy = fx  # 픽셀이 정방형(square pixel)이라고 가정
            cx, cy = w / 2.0, h / 2.0
            self.camera_intrinsics = (fx, fy, cx, cy, w, h)
            rospy.loginfo(f"Camera intrinsics cached for frame '{self.camera_frame}': fx={fx:.2f}, fy={fy:.2f}")

    def generate_and_publish(self, event=None):
        if self.latest_image_msg is None or self.camera_frame is None or self.camera_intrinsics is None:
            rospy.logwarn("Waiting for camera info...")
            return

        try:
            # [수정] 변수 네이밍 교정: lookup_transform은 target->source 변환을 반환
            cam_pos, R_map_to_cam_optical = self._lookup_camera_pose_in_map()
            # 월드->카메라 변환을 위한 회전행렬 (R_wc = R_cw^T)
            R_world_to_cam_optical = R_map_to_cam_optical.T
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        # 디버그 로그 (필요시 주석 해제)
        # self._debug_nearest_objects(cam_pos, R_world_to_cam_optical)

        visible_objects = self._get_visible_objects_advanced(cam_pos, R_world_to_cam_optical)
    

        if not visible_objects:
            return

        partial_sg = self._create_partial_sg_json(visible_objects, cam_pos, R_map_to_cam_optical)
        
        # 발행 및 저장
        self.pose_pub.publish(self._create_pose_stamped_from_tf(cam_pos, R_map_to_cam_optical))
        self.image_pub.publish(self.latest_image_msg)
        self.partial_sg_pub.publish(String(data=json.dumps(partial_sg, indent=2)))

    def _is_in_vfov(self, point, cam_pos):
        """객체의 한 점이 수직 시야각(VFOV) 120도 내에 있는지 확인"""
        # VFOV 120도는 수평면 기준 위/아래 60도를 의미
        max_angle_rad = np.deg2rad(120.0 / 2.0)
        
        vec_to_point = point - cam_pos
        dist = np.linalg.norm(vec_to_point)
        if dist < 1e-6:
            return True # 점이 카메라 위치와 동일하면 보인다고 가정

        # z축 차이를 이용해 수평면과의 각도(elevation) 계산
        # sin(theta) = opposite / hypotenuse = delta_z / dist
        elevation_angle = np.arcsin(vec_to_point[2] / dist)
        
        return abs(elevation_angle) <= max_angle_rad

    def _get_visible_objects_advanced(self, cam_pos, R_world_to_cam_optical):
        """[360도 수정] VFOV와 구면 가려짐을 기반으로 보이는 객체를 찾습니다."""
        visible = []
        
        # VFOV를 통과하는 객체들만 1차로 필터링
        candidate_objects = [obj for obj in self.all_objects if "bbox_min" in obj and self._is_in_vfov(np.array(obj["center"]), cam_pos)]

        for obj_i in candidate_objects:
            corners = self._get_bbox_corners(obj_i)
            any_corner_visible = False

            for corner_pt in corners:
                # 이 코너가 VFOV 안에 있는지 추가 확인 (중심점이 VFOV 안에 있어도 코너는 밖에 있을 수 있음)
                if not self._is_in_vfov(corner_pt, cam_pos):
                    continue

                is_occluded = False
                dist_to_corner = np.linalg.norm(corner_pt - cam_pos)
                ray_dir = (corner_pt - cam_pos) / (dist_to_corner + 1e-9)

                # 다른 후보 객체들로만 가려짐 테스트 수행
                for obj_j in candidate_objects:
                    if obj_i.get("object_id") == obj_j.get("object_id"): continue
                    if self._is_ignorable_occluder(obj_j): continue

                    intersection_dist = self._check_ray_bbox_intersection(cam_pos, ray_dir, obj_j)
                    if intersection_dist is not None and 1e-4 < intersection_dist < dist_to_corner - 1e-4:
                        is_occluded = True
                        break
                
                if not is_occluded:
                    any_corner_visible = True
                    break # 한 코너라도 보이면 이 객체는 보이는 것으로 확정
            
            if any_corner_visible:
                visible.append(obj_i)
                
        return visible

    def _lookup_camera_pose_in_map(self):
        # "map" 프레임에 대한 "camera_frame"의 변환을 조회 (카메라의 월드 상 위치/자세)
        transform = self.tf_buffer.lookup_transform("map", self.camera_frame, rospy.Time(0), rospy.Duration(0.5))
        t = transform.transform.translation
        q = transform.transform.rotation
        pos = np.array([t.x, t.y, t.z])
        # R_map_to_cam_optical: 카메라 좌표계의 축을 map 좌표계 기준으로 표현한 회전행렬
        R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        return pos, R

    def _project_world_to_pixel(self, Pw, cam_pos, R_world_to_cam_optical):
        fx, fy, cx, cy, W, H = self.camera_intrinsics
        # 월드 좌표점(Pw)을 카메라 좌표계로 변환: Pc = R_wc * (Pw - C)
        Pc = R_world_to_cam_optical.dot(Pw - cam_pos)
        if Pc[2] <= 1e-6: return False, None # 카메라 뒤
        
        u = fx * (Pc[0] / Pc[2]) + cx
        v = fy * (Pc[1] / Pc[2]) + cy
        
        return (0 <= u < W) and (0 <= v < H), (u, v, Pc[2])

    @staticmethod
    def _get_bbox_corners(obj):
        mn, mx = np.array(obj["bbox_min"]), np.array(obj["bbox_max"])
        return [np.array([x, y, z]) for x in [mn[0], mx[0]] for y in [mn[1], mx[1]] for z in [mn[2], mx[2]]]

    @staticmethod
    def _is_ignorable_occluder(obj):
        label = (obj.get("raw_label") or "").lower().replace("-", " ").strip()
        if label in STRUCTURE_LABELS: return True
        
        size = np.array(obj.get("size", [0,0,0]))
        if np.min(size) < THICKNESS_EPSILON: return True
        
        return False

    def _create_partial_sg_json(self, visible_objects, cam_pos, R_map_to_cam_optical):
        visible_object_ids = {str(obj.get("object_id")) for obj in visible_objects}
        partial_relationships = []
        full_relationships = self.full_scene_graph.get("regions", {}).get("0", {}).get("relationships", {})
        for rel_type, rel_dict in full_relationships.items():
            for subj_id, obj_id_list in rel_dict.items():
                if str(subj_id) in visible_object_ids:
                    for obj_id in obj_id_list:
                        if str(obj_id) in visible_object_ids:
                            partial_relationships.append([subj_id, rel_type, obj_id])
        
        q = tft.quaternion_from_matrix(np.vstack([np.hstack([R_map_to_cam_optical, [[0],[0],[0]]]), [0,0,0,1]]))
        return {
            "scene_name": self.scene_name,
            "timestamp_ros": rospy.Time.now().to_sec(),
            "robot_pose": {"position": cam_pos.tolist(), "orientation_xyzw": q.tolist()},
            "objects": visible_objects,
            "relationships": partial_relationships
        }

    def _create_pose_stamped_from_tf(self, position, rotation_matrix):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position = Point(*position)
        q = tft.quaternion_from_matrix(np.vstack([np.hstack([rotation_matrix, [[0],[0],[0]]]), [0,0,0,1]]))
        msg.pose.orientation = Quaternion(*q)
        return msg

    def _load_full_scene_graph(self):
        filepath = os.path.join(self.scene_files_path, f"{self.scene_name}_scene_graph.json")
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except Exception as e:
            rospy.logerr(f"Error loading scene graph file: {e}"); return None

    def _extract_all_objects(self):
        if not self.full_scene_graph: return []
        objects = []
        for region in self.full_scene_graph.get("regions", {}).values():
            objects.extend(region.get("objects", []))
        for obj in objects:
            # [수정] bbox 꼭짓점이 있으면 우선 사용, 없으면 center/size로 계산
            if "bbox" in obj and len(obj["bbox"]) == 8:
                pts = np.array(obj["bbox"])
                mn, mx = pts.min(axis=0), pts.max(axis=0)
            elif "size" in obj and "center" in obj:
                s, c = np.array(obj["size"]) / 2.0, np.array(obj["center"])
                mn, mx = c - s, c + s
            else:
                continue
            obj["bbox_min"], obj["bbox_max"] = mn.tolist(), mx.tolist()
        return objects

    def _save_partial_sg_to_file(self, data, ts):
        filepath = os.path.join(self.save_dir_sg, f"partial_sg_{ts}.json")
        try:
            with open(filepath, 'w') as f: json.dump(data, f, indent=2)
        except Exception as e:
            rospy.logerr(f"Failed to save partial SG: {e}")

    def _save_image_to_file(self, ts):
        if self.latest_image_msg is None:
            rospy.logwarn("No image message to save.")
            return

        msg = self.latest_image_msg
        filepath = os.path.join(self.save_dir_img, f"image_{ts}.png")
        try:
            enc = (msg.encoding or "bgr8").lower()
            h, w = msg.height, msg.width
            data = np.frombuffer(msg.data, dtype=np.uint8)
            step = msg.step  # bytes per row

            def rows_to_hwcn(arr_flat, channels):
                # row padding(step)을 고려해 안전하게 리셰이프
                if step == w * channels:
                    return arr_flat.reshape(h, w, channels)
                else:
                    arr2d = arr_flat.reshape(h, step)
                    arr2d = arr2d[:, :w * channels]
                    return arr2d.reshape(h, w, channels)

            if enc in ("bgr8", "rgb8"):
                ch = 3
                arr = rows_to_hwcn(data, ch)
                if enc == "bgr8":
                    arr = arr[..., ::-1].copy()  # BGR→RGB
                img = PILImage.fromarray(arr, mode="RGB")

            elif enc in ("bgra8", "rgba8"):
                ch = 4
                arr = rows_to_hwcn(data, ch)
                if enc == "bgra8":
                    arr = arr[..., [2, 1, 0, 3]].copy()  # BGRA→RGBA
                img = PILImage.fromarray(arr, mode="RGBA").convert("RGB")  # 알파 제거

            elif enc in ("mono8", "8uc1"):
                expected = h * step
                arr2d = data[:expected].reshape(h, step)[:, :w]
                img = PILImage.fromarray(arr2d, mode="L").convert("RGB")

            elif enc in ("16uc1", "mono16"):
                data16 = np.frombuffer(msg.data, dtype=np.uint16)
                # step는 2바이트 단위 * w 이어야 함. padding 고려:
                row_words = step // 2
                arr2d = data16.reshape(h, row_words)[:, :w]
                # 간단히 16비트를 8비트로 축소(시각화용)
                arr8 = np.clip(arr2d / 256.0, 0, 255).astype(np.uint8)
                img = PILImage.fromarray(arr8, mode="L").convert("RGB")

            else:
                raise ValueError(f"Unsupported encoding: {enc}")

            img.save(filepath, format="PNG")
        except Exception as e:
            rospy.logerr(f"Failed to save image via Pillow-only path: {e}")

    @staticmethod
    def _check_ray_bbox_intersection(ray_origin, ray_direction, obj):
        if "bbox_min" not in obj: return None
        bbox_min, bbox_max = np.array(obj["bbox_min"]), np.array(obj["bbox_max"])
        t_min, t_max = 0.0, float('inf')
        for i in range(3):
            if abs(ray_direction[i]) < 1e-6:
                if ray_origin[i] < bbox_min[i] or ray_origin[i] > bbox_max[i]: return None
            else:
                t1 = (bbox_min[i] - ray_origin[i]) / ray_direction[i]
                t2 = (bbox_max[i] - ray_origin[i]) / ray_direction[i]
                if t1 > t2: t1, t2 = t2, t1
                t_min, t_max = max(t_min, t1), min(t_max, t2)
                if t_min > t_max: return None
        return t_min if t_min > 1e-6 else None

    def _debug_nearest_objects(self, cam_pos, R_world_to_cam_optical, k=10):
        sorted_objs = sorted(self.all_objects, key=lambda o: np.linalg.norm(np.array(o.get("center", [0,0,0])) - cam_pos))
        for o in sorted_objs[:k]:
            if "center" not in o: continue
            Pw = np.array(o["center"])
            ok, uvz = self._project_world_to_pixel(Pw, cam_pos, R_world_to_cam_optical)
            if ok:
                u, v, z = uvz
                rospy.loginfo(f"[DBG] id={o.get('object_id')} {o.get('raw_label', ''):<15} -> u={u:.1f}, v={v:.1f}, z={z:.2f}m [IN_FOV]")
            else:
                rospy.loginfo(f"[DBG] id={o.get('object_id')} {o.get('raw_label', ''):<15} -> [OUT_OF_FOV]")
        rospy.loginfo("------------------------------------")

if __name__ == "__main__":
    try:
        node = PartialSceneGraphGenerator()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unhandled exception in PartialSceneGraphGenerator: {e}")
        import traceback
        traceback.print_exc()


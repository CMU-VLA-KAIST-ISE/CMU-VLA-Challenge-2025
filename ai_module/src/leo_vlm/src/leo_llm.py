#!/opt/conda/envs/leo/bin/python
# -*- coding: utf-8 -*-

import os, sys, json, re, math, glob, shutil
from datetime import datetime

# -------- project paths (leo_vlm root + embodied_generalist) --------
FILE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
EG_ROOT   = os.path.join(PROJ_ROOT, "embodied_generalist")
for p in (PROJ_ROOT, EG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


from hydra import initialize_config_dir, compose
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

import embodied_generalist.inference_brief as eg_infer
from embodied_generalist.common.misc import rgetattr
import embodied_generalist.common.io_utils as iu

import rospy
from std_msgs.msg import String, Int32
import torch
import threading

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False) # Gradient 불가능하도록
class LeoLLMNode(object):
    """
    /user_query(자연어 질문)->self.probe.forward , /partial_sg(JSON) -> 저장
    이미지의 경우 CLIP? 등 특정 임베딩 차원으로 옮겨야 함..
    답변은 /target_obj_idx(Int32) 및 /llm_response(String) 발행
    """
    def __init__(self):
        self.topic_partial_sg = "/partial_scene_graph_generator/partial_scene_graph"
        #self.topic_partial_sg = "/partial_sg"
        self.topic_user_query = "/user_query"
        self.topic_target_idx = "/target_obj_idx"
        self.topic_llm_resp   = "/llm_response"
        self.topic_visted_list = "/visited_list"

        self.enable_logs = bool(int(rospy.get_param("~enable_logs", 1)))

        self.last_partial_sg = None
        self.last_query      = None
        self.visited_list = None

        self.edge_ths = 10

        # Pub/Sub
        self.pub_target = rospy.Publisher(self.topic_target_idx, Int32, queue_size=10, latch=True)
        self.pub_resp   = rospy.Publisher(self.topic_llm_resp,   String, queue_size=10, latch=True)
        rospy.Subscriber(self.topic_partial_sg, String, self.cb_partial_sg, queue_size=1)
        rospy.Subscriber(self.topic_user_query, String, self.cb_user_query, queue_size=1)
        rospy.Subscriber(self.topic_visted_list, String, self.cb_visited_list, queue_size=1)
        # rospy.Subscriber("/scene_image", Image, self.cb_image, queue_size=1)  # 필요시
        
        
        config_dir = os.path.join(EG_ROOT, "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="default",
            overrides=[
                "vision3d=ose3d_pointnetpp",
            ])
        OmegaConf.set_struct(cfg, False)

        print("project name:", cfg.name)
        cfg.base_dir = to_absolute_path(cfg.base_dir)
        naming_keys = [cfg.name]
        for name in cfg.naming_keywords:
            key = str(rgetattr(cfg, name))
            if key: naming_keys.append(key)
        exp_name = "_".join(naming_keys)
        cfg.exp_dir = os.path.join(
            cfg.base_dir, exp_name,
            datetime.now().strftime("%Y-%m-%d-%H:%M:%S") if "time" in cfg.naming_keywords else ""
        )
        cfg.exp_dir = to_absolute_path(cfg.exp_dir)
        iu.make_dir(cfg.exp_dir)
        
        def _resolve_ckpt(p_raw: str) -> str:
            if os.path.isabs(p_raw):
                candidates = [p_raw]
            else:
                candidates = [
                    os.path.join(EG_ROOT, p_raw),
                    os.path.join(PROJ_ROOT, p_raw),
                    to_absolute_path(p_raw),
                ]

            for c in candidates:
                if os.path.isdir(c):
                    return c
                if os.path.isfile(c):
                    return c
            raise FileNotFoundError(f"Checkpoint not found. Tried: {candidates}")
        
        cfg.pretrained_ckpt_path = "results/sft_noact"
        cfg.pretrained_ckpt_path = _resolve_ckpt(str(cfg.pretrained_ckpt_path))

        mask3d_default = os.path.join(EG_ROOT, "Mask3D_data")
        scb = getattr(cfg, "data.scan_family_base", None)
        if not scb:
            cfg.data.scan_family_base = mask3d_default
        else:
            if not os.path.isabs(scb):
                cand1 = os.path.join(EG_ROOT, scb)
                cand2 = os.path.join(PROJ_ROOT, scb)
                if os.path.exists(cand1):
                    cfg.data.scan_family_base = cand1
                elif os.path.exists(cand2):
                    cfg.data.scan_family_base = cand2
                else:
                    cfg.data.scan_family_base = to_absolute_path(scb)

        print("scannet_base:", cfg.data.scan_family_base)
        print("exp_dir : ", cfg.exp_dir)
        print("pretrained_ckpt:", cfg.pretrained_ckpt_path)
        

        def _as_list(x):
            if isinstance(x, list): return x
            if x is None: return []
            return [x]
        srcs = _as_list(cfg.probe.get("sources"))
        scids = _as_list(cfg.probe.get("scene_ids"))
        sits = _as_list(cfg.probe.get("situations"))
        instr = _as_list(cfg.probe.get("instructions"))
        N = max(len(srcs), len(scids), len(sits), len(instr), 1)
        def _pad(lst, default):
            if len(lst) == 0: return [default]*N
            if len(lst)  < N: return lst + [lst[-1]]*(N-len(lst))
            return lst[:N]

        cfg.probe.sources      = _pad(srcs,  "mask3d")
        cfg.probe.scene_ids    = _pad(scids, "")
        cfg.probe.situations   = _pad(sits,  "")
        cfg.probe.instructions = _pad(instr, "Describe this scene.")

        eg_infer.cfg = cfg
        self.prober = eg_infer.LeoProber(cfg)

        # 여기서부터 LEO의 OOM 문제의 해결을 위해 추가한 코드입니다. (최원혁)
        try:
            self.prober.model.eval()
        except Exception:
            pass
        
        self._lock = threading.Lock()
        self._busy = False
        
        pcd_proj = getattr(self.prober.model, "pcd_proj", None)
        assert pcd_proj is not None, "model.pcd_proj가 없습니다."
        try:
            in_dim = pcd_proj.in_features
        except AttributeError:
            in_dim = pcd_proj.weight.shape[1]
        self._B, self._O = 1, 1
        self.obj_tokens_buf = torch.zeros(self._B, self._O, in_dim, device=device)
        self.obj_masks_buf  = torch.ones (self._B, self._O,          dtype=torch.bool, device=device)
        self.obj_locs_buf   = torch.zeros(self._B, self._O, 3,       device=device)
        self.anchor_buf     = torch.zeros(1, 3, device=device)
        self.img_fts_buf    = torch.zeros(1, 3, 224, 224, device=device)
        self.img_masks_buf  = torch.zeros(1, 1,            dtype=torch.bool, device=device)
        
        # 여기까지
        rospy.loginfo("[LeoLLM] ready. ckpt=%s, scannet_base=%s",
                      cfg.pretrained_ckpt_path, cfg.data.scan_family_base)
    
    def cb_visited_list(self, msg:String):
        try:
            self.visited_list = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("[LeoLLM] bad visited list JSON: %s", e)
            return
    
    def cb_partial_sg(self, msg: String):
        try:
            self.last_partial_sg = self.preprocess_sg(json.loads(msg.data))
        except Exception as e:
            rospy.logwarn("[LeoLLM] bad partial_sg JSON: %s", e)
            return
    
    def preprocess_sg(self, partial_sg:dict):
        # Preassumption : partial sg is the sg that only includes objects visible at the current position
        # attributes to use : objects.[object_id, raw_label, center, ]  relationships[ : ths],  robot_pose (if exist)
        preprocess_partiral_sg = {"objects":[], "relationships":[]}

        nodes = partial_sg.get("objects", []) or []
        edges = partial_sg.get("relationships", []) or []

        for node in nodes:
            pre_node = {
                "object_id" : node.get("object_id", None),   # can be used as int format
                "raw_label" : node.get("raw_label", None), 
                "center" : node.get("center", None),
            }
            preprocess_partiral_sg["objects"].append(pre_node)
        
        preprocess_partiral_sg["relationships"] = edges[:self.edge_ths]
        preprocess_partiral_sg["robot_pose"] = partial_sg.get("robot_pose", None)
        preprocess_partiral_sg["scene_name"] = partial_sg.get("scene_name", None)

        return preprocess_partiral_sg

    def cb_user_query(self, msg: String):
        self.last_query = msg.data.strip()
        self.leo_infer()

    # def cb_image(self, msg: Image):
    #     image should be encoded as the pre defined model (convnext)
    #     pass

    def leo_infer(self):
        if not self.last_partial_sg or not self.last_query:
            return
        # 동시 실행 방지를 위해 추가로 추가한 코드입니다 (최원혁)
        if not self._lock.acquire(blocking=False):
            rospy.logwarn("[LeoLLM] inference busy, skipping")
            return
        if self.busy:
            self._lock.release()
            return
        self._busy=True
        # 여기까지
        
        data_dict = self.build_data_dict(self.last_query, self.last_partial_sg)
        # idx, raw = self.run_prober(data_dict)
        # 여기도 아래처럼 변경
        try:
            idx, raw = self.run_prober(data_dict)
        finally:
            self._busy = False
            self._lock.realease()
        print(f"idx as {idx}")
        print(f"response as {raw}")
        
        try :
            target_idx = int(idx)
        except Exception as e:
            print(f"no integer, selected as default index. error as : {e}")
            target_idx = 0

        self.pub_target.publish(Int32(data=target_idx))
        self.pub_resp.publish(String(data=str(raw)))
        rospy.loginfo("[LeoLLM] target_obj_idx=%s", idx)

    def build_data_dict(self, question: str, partial_sg: dict) -> str:
        r""" Unified input format:
        <prompt_before_obj> + <prompt_middle_1> + <img_tokens> + <prompt_middle_2> + <obj_tokens> + <prompt_after_obj>
        <prompt_before_obj>: <role_prompt> + <situation_prompt>
        <prompt_middle_1>: <egoview_prompt> (masked if unnecessary)
        <prompt_middle_2>: <objects_prompt>
        <prompt_after_obj>: <task_prompt>
        <output_gt>: response label, will be appended to input sequence for computing loss during training
        """

        role_prompt = "You are an AI visual assistant situated in a 3D scene. "\
                    "You can perceive (1) the objects (including yourself) in the scene graph (always accessible). "\
                    "You should properly respond to the USER's instruction according to the given visual information."
        
        situation_prompt = f"You are at a selected location in the 3D scene."
        #egoview_prompt = "Ego-view image:"
        egoview_prompt = ""
        objects_prompt = "Objects (including you) and their relationships in the scene are as :"
        instruction = f"""find the relatively closest objects and return it idx as integer (like 0, 1, 2, ..) or only noun (as chair, tree, ..) for given question {question}.
          Please consider already visited objects to determine whether you are solved the question or not. If you solved the question please return -1."""
        task_prompt = f"USER: {instruction} ASSISTANT:"

        if self.visited_list is not None:
            visited_lines = ["already visited objects are as"]
            for label, position in self.visited_list:
                visited_lines.append(f"{label} at {position},")
            visited_sum = "\n".join(visited_lines).rstrip(',')
        else:
            visited_sum = "there are no visited objects"

        scene_name = partial_sg["scene_name"]
        nodes = partial_sg["objects"]
        edges = partial_sg["relationships"]

        lines = []

        for node in nodes:
            id = node["object_id"]
            label = node["raw_label"]
            lines.append(f"object {id} : {label}, ")
        
        for edge in edges:
            lines.append(f"predicate : object {edge[0]} is {edge[1]} to object {edge[2]}, ")
        
        summary = "\n".join(lines)

        # 위의 init에서 선언
        """pcd_proj = getattr(self.prober.model, "pcd_proj", None)
        assert pcd_proj is not None, "model.pcd_proj가 없습니다."
        
        try:
            in_dim = pcd_proj.in_features
        except AttributeError:
            in_dim = pcd_proj.weight.shape[1]
        
        B, O, P = 1, 1, 1"""

        data_dict = {
            'source': "scene graph",
            'scene_id': scene_name,
            'prompt_before_obj': [role_prompt + situation_prompt],
            'prompt_middle_1': [egoview_prompt],
            'prompt_middle_2': [objects_prompt + summary + visited_sum],
            'prompt_after_obj': [task_prompt],
            'obj_tokens': self.obj_tokens_buf,
            'obj_masks':  self.obj_masks_buf,
            'obj_locs':   self.obj_locs_buf,
            'anchor_locs': self.anchor_buf,
            'img_fts':     self.img_fts_buf,
            'img_masks':   self.img_masks_buf,
            }

        return data_dict

    def run_prober(self, data_dict: dict):
        try:
            with torch.inference_mode():
                out = self.prober.forward(data_dict, inference=True)
            txt = None
            rospy.loginfo("data_dict : %s",data_dict)
            if isinstance(out, dict) and "output_txt" in out and len(out["output_txt"]) > 0:
                txt = out["output_txt"][0]
                
            del out
            return self.parse_idx(txt), txt
        except Exception as e:
            rospy.logwarn("[LeoLLM] forward path failed: %s", e)
            
        return None, None
    
    def parse_idx(self, response):
        self.idx_parser = idx_parser(self.last_partial_sg)
        if response is None:
            return None
        if isinstance(response, int):
            return response
        if isinstance(response, dict) and "target_obj_idx" in response:
            try: return int(response["target_obj_idx"])
            except Exception: return None
        if isinstance(response, str):
            s = response.strip()
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
            m = re.search(r"-?\d+", s)
            if m: return int(m.group(0))
            psuedo_idx = self.idx_parser.psuedo_idx_sg(response)
            return psuedo_idx

        return None


import re

class idx_parser:
    def __init__(self, partial_sg):
        self.last_partial_sg = partial_sg
    
    def _norm(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _tokens(self, s: str):
        STOP = {"the","a","an","this","that","these","those","my","your","his","her","its","our","their",
                "to","of","and","or","is","are","was","were","be","been","being","there","here","it","on",
                "as","for","at","in","into","with","by"}
        return [t for t in self._norm(s).split() if t and t not in STOP]

    def _subject_span(self, s: str) -> str:
        text = self._norm(s)
        PREP_PATTERNS = [
            r"in\s+front\s+of", r"next\s+to", r"close\s+to", r"near\s+by",
            r"between", r"among", r"beside", r"behind", r"under", r"below",
            r"above", r"over", r"around", r"across", r"along", r"toward", r"towards",
            r"near", r"by", r"with", r"on", r"in", r"at", r"from", r"of"
        ]
        cut = len(text)
        for pat in PREP_PATTERNS:
            m = re.search(r"\b" + pat + r"\b", text)
            if m and m.start() < cut:
                cut = m.start()
        return text[:cut].strip()

    def _label_synonyms(self, raw_label: str):
        CANON = {
            "sofa": {"sofa","couch","settee"},
            "tv": {"tv","television","screen","monitor"},
            "trash can": {"trash","trashcan","garbage","bin","garbage bin","wastebasket","trash can"},
            "potted plant": {"plant","potted plant","houseplant","flowerpot"},
            "lamp": {"lamp","light","ceiling lamp","floor lamp","desk lamp","light fixture","spotlight","focus light"},
            "table": {"table","desk"},
            "chair": {"chair","seat","stool"},
            "vase": {"vase","pot","jar"},
            "shelf": {"shelf","shelving","bookcase","bookshelf"},
        }
        rl = self._norm(raw_label)
        for canon, syns in CANON.items():
            if any(x in rl for x in syns | {canon}):
                return {self._norm(x) for x in syns | {canon}}
        return {" ".join(self._tokens(rl))} | set(self._tokens(rl))

    def psuedo_idx_sg(self, response: str) -> int:
        nodes = (self.last_partial_sg or {}).get("objects", []) or []
        if not nodes:
            return 0

        subj = self._subject_span(response)              # 예: "potted plant"
        subj_tokens = set(self._tokens(subj))            # {"potted","plant"}

        best_idx, best_score = 0, -1
        for node in nodes:
            raw_label = node.get("raw_label") or node.get("nyu40_label") or ""
            syn_phrases = self._label_synonyms(raw_label)  # {"potted plant","plant",...}

            score = 0
            for phrase in syn_phrases:
                if not phrase: 
                    continue
                # 연속 구문으로 들어있으면 보너스
                if re.search(r"\b" + re.escape(phrase) + r"\b", subj):
                    score += 3 + phrase.count(" ")  # 멀티워드일수록 가산

            label_tokens = set()
            for phrase in syn_phrases:
                label_tokens |= set(self._tokens(phrase))
            token_overlap = len(subj_tokens & label_tokens)
            score += token_overlap

            if score == 0:
                resp_tokens = set(self._tokens(response))
                score += 0.5 * len(resp_tokens & label_tokens)

            if score > best_score:
                idx = node.get("object_id")
                best_idx, best_score = idx, score
            
        return int(best_idx)


def main():
    rospy.init_node("leo_llm", anonymous=False)
    node = LeoLLMNode()
    rospy.loginfo("leo_llm node started.")
    rospy.spin()


if __name__ == "__main__":
    main()

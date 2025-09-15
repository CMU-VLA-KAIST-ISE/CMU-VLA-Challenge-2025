#!/opt/conda/envs/leo/bin/python
# -*- coding: utf-8 -*-

import os, json, re
import rospy
from std_msgs.msg import String, Int32
from openai import OpenAI
class GPTIdxParser:
    STOP = {"the","a","an","this","that","these","those","my","your","his","her","its",
            "our","their","to","of","and","or","is","are","was","were","be","been",
            "being","there","here","it","on","as","for","at","in","into","with","by"}
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
    def __init__(self, partial_sg):
        self.partial_sg = partial_sg or {}
    def _norm(self, s):
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    def _tokens(self, s):
        return [t for t in self._norm(s).split() if t and t not in self.STOP]
    def _subject_span(self, s):
        text = self._norm(s)
        PREP = [r"in\s+front\s+of", r"next\s+to", r"close\s+to", r"near\s+by",
                r"between", r"among", r"beside", r"behind", r"under", r"below",
                r"above", r"over", r"around", r"across", r"along", r"toward", r"towards",
                r"near", r"by", r"with", r"on", r"in", r"at", r"from", r"of"]
        cut = len(text)
        for pat in PREP:
            m = re.search(r"\b" + pat + r"\b", text)
            if m and m.start() < cut:
                cut = m.start()
        return text[:cut].strip()
    def _label_synonyms(self, raw_label):
        rl = self._norm(raw_label or "")
        for canon, syns in self.CANON.items():
            if any(x in rl for x in syns | {canon}):
                return {self._norm(x) for x in syns | {canon}}
        toks = set(self._tokens(rl))
        return {" ".join(toks)} | toks
    def choose_idx(self, response_text):
        nodes = (self.partial_sg or {}).get("objects", []) or []
        if not nodes:
            return 0
        subj = self._subject_span(response_text or "")
        subj_tokens = set(self._tokens(subj))
        best_idx, best_score = 0, -1
        for node in nodes:
            raw_label = node.get("raw_label") or node.get("nyu40_label") or ""
            syn_phrases = self._label_synonyms(raw_label)
            score = 0
            for phrase in syn_phrases:
                if phrase and re.search(r"\b" + re.escape(phrase) + r"\b", subj):
                    score += 3 + phrase.count(" ")
            label_tokens = set()
            for phrase in syn_phrases:
                label_tokens |= set(self._tokens(phrase))
            score += len(subj_tokens & label_tokens)
            if score == 0:
                resp_tokens = set(self._tokens(response_text or ""))
                score += 0.5 * len(resp_tokens & label_tokens)
            if score > best_score:
                idx = node.get("object_id")
                best_idx, best_score = idx, score
        return int(best_idx)


class GPTLLMNode(object):
    """
    /partial_sg(JSON) + /user_query(String) -> OpenAI Responses API 호출
    결과 JSON에서 target_obj_idx / answer_text 추출 후 publish.
    """
    def __init__(self):
        # Topics (기존과 동일)
        self.topic_partial_sg = "/partial_scene_graph_generator/partial_scene_graph"
        self.topic_user_query = "/user_query"
        self.topic_target_idx = "/target_obj_idx"
        self.topic_llm_resp   = "/llm_response"

        # Params
        self.enable_logs = bool(int(rospy.get_param("~enable_logs", 1)))
        self.edge_ths    = int(rospy.get_param("~edge_ths", 10))
        # 모델명은 필요에 따라 교체 가능 (예: gpt-4o, gpt-4o-mini, gpt-5 등)
        self.model_name  = rospy.get_param("~model_name", "gpt-5")
        # State
        self.last_partial_sg = None
        self.last_query      = None
        # ROS pub/sub
        self.pub_target = rospy.Publisher(self.topic_target_idx, Int32, queue_size=10, latch=True)
        self.pub_resp   = rospy.Publisher(self.topic_llm_resp,   String, queue_size=10, latch=True)
        rospy.Subscriber(self.topic_partial_sg, String, self.cb_partial_sg, queue_size=1)
        rospy.Subscriber(self.topic_user_query, String, self.cb_user_query, queue_size=1)

        # OpenAI client (환경변수 OPENAI_API_KEY 사용 권장)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        rospy.loginfo("[GPT-LLM] ready. model=%s", self.model_name)

    # ---------- ROS Callbacks ----------
    def cb_partial_sg(self, msg: String):
        try:
            self.last_partial_sg = self.preprocess_sg(json.loads(msg.data))

        except Exception as e:
            rospy.logwarn("[GPT-LLM] bad partial_sg JSON: %s", e)

    def cb_user_query(self, msg: String):
        self.last_query = msg.data.strip()
        self.infer()

    # ---------- Helpers ----------
    def preprocess_sg(self, partial_sg: dict):
        out = {"objects": [], "relationships": []}
        nodes = (partial_sg or {}).get("objects", []) or []
        edges = (partial_sg or {}).get("relationships", []) or []
        for node in nodes:
            out["objects"].append({
                "object_id": node.get("object_id"),
                "raw_label": node.get("raw_label"),
                "center":    node.get("center"),
            })
        out["relationships"] = edges[: self.edge_ths]
        out["robot_pose"]    = (partial_sg or {}).get("robot_pose")
        out["scene_name"]    = (partial_sg or {}).get("scene_name")
        return out

    def build_prompt(self, question: str, partial_sg: dict) -> str:
        scene = partial_sg.get("scene_name")
        nodes = partial_sg.get("objects", []) or []
        edges = partial_sg.get("relationships", []) or []

        obj_lines = []
        for n in nodes:
            obj_lines.append(f"- object {n.get('object_id')}: {n.get('raw_label')}")
        rel_lines = []
        for e in edges:
            # e = [src, predicate, dst] 가정
            try:
                rel_lines.append(f"- object {e[0]} is {e[1]} to object {e[2]}")
            except Exception:
                pass

        return f"""You are an AI assistant situated in a 3D scene.
        You are given a scene graph listing objects (with indices) and their relationships.
        Your job: pick the SINGLE best-matching object index for the USER's request.
        Rules:
        - Choose by semantic match to the label (consider common synonyms).
        - If multiple match, prefer the relatively closest / most salient one implied by relationships.
        - If unsure, choose the most likely label match deterministically.
        Return ONLY a JSON object that matches the required schema (no extra keys, no commentary).

        SCENE: {scene}
        OBJECTS:
        {chr(10).join(obj_lines)}
        RELATIONSHIPS:
        {chr(10).join(rel_lines)}

        USER: {question}
        """

    def openai_call(self, prompt: str):
        """
        Responses API + Structured Outputs(JSON Schema) 사용.
        schema: { target_obj_idx:int >=0, answer_text:str }
        """
        schema = {
            "name": "scene_graph_selection",
            "schema": {
                "type": "object",
                "properties": {
                    "target_obj_idx": {"type": "integer", "minimum": 0},
                    "answer_text":    {"type": "string"}
                },
                "required": ["target_obj_idx", "answer_text"],
                "additionalProperties": False
            },
            "strict": True
        }
        try:
            """
            resp = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            # SDK가 합쳐서 제공하는 텍스트 접근자 (JSON 문자열이 도착)
            txt = resp.output_text  # e.g. '{"target_obj_idx": 2, "answer_text": "light switch"}'
            """
            chat=self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                        "content": "Return ONLY a minified JSON object with keys: target_obj_idx (integer >= 0), answer_text (string). No extra text."},
                    {"role": "user", "content": prompt}
                ]
            )
            txt = chat.choices[0].message.content

            data = json.loads(txt)
            return int(data.get("target_obj_idx")), str(data.get("answer_text", "")).strip(), txt
        except Exception as e:
            rospy.logwarn("[GPT-LLM] OpenAI call/parse failed: %s", e)
            return None, None, None

    def infer(self):
        if not self.last_partial_sg or not self.last_query:
            return
        rospy.loginfo(self.last_partial_sg)
        prompt = self.build_prompt(self.last_query, self.last_partial_sg)

        idx, answer_text, raw_json = self.openai_call(prompt)
        rospy.loginfo(answer_text)

        # 백업: 모델이 JSON을 못 지켰거나 idx가 비정상일 때
        if idx is None or idx < 0:
            parser = GPTIdxParser(self.last_partial_sg)
            fallback_idx = parser.choose_idx(answer_text or self.last_query)
            idx = fallback_idx
            if not answer_text:
                # 최소한 레이블 명사 하나 추출
                answer_text = " ".join(parser._tokens(self.last_query)) or "object"

        # Publish
        self.pub_target.publish(Int32(data=int(idx)))
        final_resp = raw_json if raw_json else json.dumps({
            "target_obj_idx": int(idx), "answer_text": answer_text
        })
        self.pub_resp.publish(String(data=final_resp))
        rospy.loginfo("[GPT-LLM] target_obj_idx=%s", idx)


def main():
    rospy.init_node("gpt_llm", anonymous=False)
    node = GPTLLMNode()
    rospy.loginfo("gpt_llm node started.")
    rospy.spin()

if __name__ == "__main__":
    main()

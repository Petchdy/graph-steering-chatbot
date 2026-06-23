"""Swappable extractors.

StubExtractor  - offline key:value regex. Used by tests.
CBTExtractor   - 4-step per-turn pipeline via Ollama /api/generate.
"""

import json
import re

from interfaces import GraphNode


class StubExtractor:
    """Looks for `<field_key>: <value>` lines in the message.

    Offline and deterministic — only used for tests.
    """

    _PATTERN = re.compile(r"(\w+)\s*[:=]\s*(.+)")

    def extract(self, message: str, schema_text: str) -> dict[str, str]:
        known_keys = set(re.findall(r"^- (\w+)", schema_text, flags=re.MULTILINE))
        deltas: dict[str, str] = {}
        for line in message.splitlines():
            m = self._PATTERN.match(line.strip())
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip()
            if key in known_keys:
                deltas[key] = value
        return deltas

    def extract_nodes(self, message: str, window: list[tuple[str, str]],
                      schema_text: str) -> list[dict]:
        return []

    def resolve_edges(self, new_node: GraphNode, existing_nodes: list[GraphNode],
                      window_text: str, subject_edges: dict) -> list[tuple[str, str, str]]:
        return []


class CBTExtractor:
    """4-step per-turn extraction pipeline via Ollama.

    A. extract_nodes(): extract + atomize → list of node candidates
    B. extract(): flat field extraction for apply_deltas() (backward compat)
    C. resolve_edges(): confirm edge presence between a new node and existing found nodes
    """

    def __init__(self, model: str = "qwen3.5-nothink", host: str = "http://localhost:11434"):
        self._model = model
        self._host = host.rstrip("/")

    def _ollama_generate(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt,
                  "stream": False, "format": "json",
                  "think": False, "keep_alive": "10m",
                  "options": {"temperature": 0}},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _parse_json(self, raw: str) -> dict | list:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def extract(self, message: str, schema_text: str) -> dict[str, str]:
        """Flat field extraction for apply_deltas() — backward compatible."""
        from prompts import CBT_EXTRACTION_PROMPT
        prompt = CBT_EXTRACTION_PROMPT.format(
            ontology_schema=schema_text,
            window="",
            message=message,
        )
        raw = self._ollama_generate(prompt)
        result = self._parse_json(raw)
        if not isinstance(result, dict):
            return {}
        known_keys = set(re.findall(r"^- (\w+)", schema_text, flags=re.MULTILINE))
        return {k: str(v) for k, v in result.items() if k in known_keys and isinstance(v, str) and v}

    def extract_nodes(self, message: str, window: list[tuple[str, str]],
                      ontology_text: str) -> list[dict]:
        """Extract rich graph node candidates from message with conversation window.

        Returns: [{"label": "Situation", "props": {...}}, ...]
        """
        from prompts import CBT_NODE_EXTRACTION_PROMPT
        window_text = "\n".join(
            f"{'Therapist' if role == 'therapist' else 'Client'}: {text}"
            for role, text in window
        ) or "(no prior turns)"
        prompt = CBT_NODE_EXTRACTION_PROMPT.format(
            ontology_text=ontology_text,
            window=window_text,
            message=message,
        )
        raw = self._ollama_generate(prompt)
        result = self._parse_json(raw)
        nodes = []
        if isinstance(result, dict) and isinstance(result.get("nodes"), list):
            nodes = result["nodes"]
        elif isinstance(result, list):
            nodes = result
        out = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            label = n.get("label")
            props = n.get("props")
            if isinstance(label, str) and isinstance(props, dict) and props:
                out.append({"label": label, "props": props})
        return out

    def resolve_edges(self, new_node: GraphNode, existing_nodes: list[GraphNode],
                      window_text: str, subject_edges: dict) -> list[tuple[str, str, str]]:
        """Confirm which edges between new_node and already-found nodes exist in the window."""
        from prompts import CBT_EDGE_RESOLUTION_PROMPT

        found_by_label: dict[str, list] = {}
        for n in existing_nodes:
            if n.status == "found" and n.node_id != new_node.node_id:
                found_by_label.setdefault(n.label, []).append(n)

        candidates = []
        for pred, obj_label in subject_edges.get(new_node.label, []):
            for obj_node in found_by_label.get(obj_label, []):
                candidates.append((new_node.node_id, pred, obj_node.node_id,
                                   new_node, obj_node))

        if not candidates:
            return []

        candidate_lines = "\n".join(
            f"{i+1}. {c[3].label}({str(c[3].props.get('content', '') or c[3].props.get('description', ''))[:40]!r}) "
            f"--[{c[1]}]--> {c[4].label}({str(c[4].props.get('content', '') or c[4].props.get('description', ''))[:40]!r})"
            for i, c in enumerate(candidates)
        )
        prompt = CBT_EDGE_RESOLUTION_PROMPT.format(
            window=window_text,
            candidates=candidate_lines,
        )
        raw = self._ollama_generate(prompt)
        result = self._parse_json(raw)
        confirmed_indices = result.get("confirmed", []) if isinstance(result, dict) else []
        return [
            (candidates[i-1][0], candidates[i-1][1], candidates[i-1][2])
            for i in confirmed_indices
            if isinstance(i, int) and 1 <= i <= len(candidates)
        ]

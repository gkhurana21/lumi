"""Scene memory: latest known location per object, semantic retrieval via Chroma's default embedder."""
from __future__ import annotations

import re
import time


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "object"


def _ago(sec: float) -> str:
    if sec < 20:
        return "just now"
    if sec < 90:
        return f"{int(sec)}s ago"
    return f"{int(sec // 60)} min ago"


class SceneMemory:
    def __init__(self, path: str = "data/chroma"):
        import chromadb
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection("scene_objects")

    def upsert_objects(self, objects: list[dict], ts: float) -> None:
        by_id = {_slug(o["name"]): o for o in objects if o.get("name")}
        if not by_id:
            return
        self.col.upsert(
            ids=list(by_id),
            documents=[f'{o["name"]}: {o.get("location", "")}. {o.get("details", "")}'.strip() for o in by_id.values()],
            metadatas=[{"name": o["name"], "location": o.get("location", ""), "last_seen": ts} for o in by_id.values()],
        )

    def search(self, query: str, k: int = 4) -> list[str]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, n))
        now = time.time()
        return [f'{m["name"]} was {m["location"]} ({_ago(now - m["last_seen"])})' for m in res["metadatas"][0]]


def _tokens(s: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", s.lower())}


class InMemorySceneMemory:
    """Dependency-free fallback: keyword overlap, ties broken by recency. For tests and offline demos."""

    def __init__(self):
        self._objs: dict[str, dict] = {}

    def upsert_objects(self, objects: list[dict], ts: float) -> None:
        for o in objects:
            if o.get("name"):
                self._objs[_slug(o["name"])] = {**o, "last_seen": ts}

    def search(self, query: str, k: int = 4) -> list[str]:
        q = _tokens(query)
        hits = [(len(q & _tokens(f'{o["name"]} {o.get("details", "")}')), o) for o in self._objs.values()]
        hits = sorted((h for h in hits if h[0]), key=lambda h: (-h[0], -h[1]["last_seen"]))
        now = time.time()
        return [f'{o["name"]} was {o.get("location", "")} ({_ago(now - o["last_seen"])})' for _, o in hits[:k]]


def make_memory(kind: str, path: str):
    return SceneMemory(path) if kind == "chroma" else InMemorySceneMemory()

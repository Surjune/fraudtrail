"""Connection to TigerGraph Savanna.

Thin wrapper over pyTigerGraph: connect, run GSQL, run installed queries, upload loading
files. Errors are raised as `GraphError` so callers never see a raw HTTP failure, and a
failed call never returns partial or invented data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fraudtrail.config import Settings, TigerGraphSettings, load_settings

log = logging.getLogger(__name__)


class GraphError(RuntimeError):
    """A TigerGraph call failed or returned something unusable."""


class GraphClient:
    def __init__(self, settings: TigerGraphSettings) -> None:
        settings.require()
        self._settings = settings
        self._conn = self._connect(settings)

    @classmethod
    def from_env(cls, settings: Settings | None = None) -> GraphClient:
        return cls((settings or load_settings()).tigergraph)

    @staticmethod
    def _connect(s: TigerGraphSettings) -> Any:
        try:
            import pyTigerGraph as tg
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise GraphError("pyTigerGraph is not installed") from exc
        conn = tg.TigerGraphConnection(
            host=s.host,
            graphname=s.graph,
            username=s.username or "tigergraph",
            password=s.password,
            restppPort=s.restpp_port,
            gsPort=s.gs_port,
        )
        try:
            if s.secret:
                conn.getToken(s.secret)
            else:
                conn.getToken()
        except Exception as exc:
            raise GraphError(f"could not authenticate to {s.host}: {exc}") from exc
        return conn

    @property
    def graph_name(self) -> str:
        return self._settings.graph

    def gsql(self, script: str) -> str:
        """Run a GSQL script. Raises when TigerGraph reports a semantic error."""
        try:
            result = self._conn.gsql(script)
        except Exception as exc:
            raise GraphError(f"GSQL failed: {exc}") from exc
        text = result if isinstance(result, str) else str(result)
        lowered = text.lower()
        if "semantic check fails" in lowered or "failed to create" in lowered:
            raise GraphError(f"GSQL reported an error:\n{text}")
        return text

    def run_file(self, path: Path) -> str:
        log.info("running %s", path.name)
        return self.gsql(path.read_text(encoding="utf-8"))

    def run_query(self, name: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Run an installed query and return its result list."""
        try:
            result = self._conn.runInstalledQuery(name, params or {})
        except Exception as exc:
            raise GraphError(f"query {name} failed: {exc}") from exc
        if not isinstance(result, list):
            raise GraphError(f"query {name} returned {type(result).__name__}, expected a list")
        return result

    def upload(self, job: str, tag: str, path: Path) -> Any:
        """Post one CSV to a loading job's file tag."""
        try:
            return self._conn.runLoadingJobWithFile(str(path), tag, job, sep=",")
        except Exception as exc:
            raise GraphError(f"loading {path.name} into {job}/{tag} failed: {exc}") from exc

    def vertex_counts(self) -> dict[str, int]:
        try:
            counts = self._conn.getVertexCount("*")
        except Exception as exc:
            raise GraphError(f"could not read vertex counts: {exc}") from exc
        return dict(counts) if isinstance(counts, dict) else {}

    def edge_counts(self) -> dict[str, int]:
        try:
            counts = self._conn.getEdgeCount("*")
        except Exception as exc:
            raise GraphError(f"could not read edge counts: {exc}") from exc
        return dict(counts) if isinstance(counts, dict) else {}

    def upsert_vertex(self, vertex_type: str, vertex_id: str, attributes: dict[str, Any]) -> int:
        try:
            return int(self._conn.upsertVertex(vertex_type, vertex_id, attributes))
        except Exception as exc:
            raise GraphError(f"upsert of {vertex_type} {vertex_id} failed: {exc}") from exc

    def upsert_edges(
        self,
        source_type: str,
        edge_type: str,
        target_type: str,
        pairs: list[tuple[str, str]],
    ) -> int:
        if not pairs:
            return 0
        payload: list[tuple[str, str, dict[str, Any]]] = [
            (source, target, {}) for source, target in pairs
        ]
        try:
            return int(self._conn.upsertEdges(source_type, edge_type, target_type, payload))
        except Exception as exc:
            raise GraphError(f"upsert of {edge_type} edges failed: {exc}") from exc

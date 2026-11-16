"""HTTP server run inside each model worker process.

A worker process hosts exactly one model version.  It never talks to the
registry directly: the API process owns lifecycle decisions and addresses the
worker over its private loopback HTTP port.  This keeps the worker free of
shared-state races and makes "the process is still working" observable
independently of what any HTTP client (or the API itself) is doing.

Every inference request is registered in the worker's own in-flight table —
keyed by the ``x-qc-request-id`` header the API assigns — *before* the
request is dispatched, and removed only when the computation is really over.
``GET /status`` exposes that table, so after an API↔worker connection break
the API can reconcile: a request the worker still lists is still running; a
request the worker no longer lists is finished (or never arrived).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models import Model, load_model

REQUEST_ID_HEADER = "x-qc-request-id"
SEQ_HEADER = "x-qc-seq"


class InferRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector, length must match the model")
    slow_ms: int = Field(
        0,
        ge=0,
        le=600_000,
        description="Artificial delay (demo hook) to exercise slow inference / timeouts",
    )
    drop_response: bool = Field(
        False,
        description=(
            "Demo hook: accept the request, then break the HTTP response "
            "connection immediately while the computation (slow_ms) continues "
            "in the background. Simulates an API↔worker link failure mid-"
            "inference; the request stays in the worker's in-flight table "
            "until the computation really finishes."
        ),
    )


class InferResponse(BaseModel):
    version: str
    label: str
    score: float
    inference_id: str
    elapsed_ms: float


class InflightEntry(BaseModel):
    request_id: str
    seq: int
    elapsed_ms: float


class StatusResponse(BaseModel):
    version: str
    slot_id: str
    pid: int
    in_flight: int
    accepted: int
    completed: int
    in_flight_requests: List[InflightEntry]
    max_seen_seq: int
    started_at: float
    uptime_seconds: float
    n_features: int
    n_classes: int
    labels: List[str]


class _InflightRegistry:
    """Thread-safe table of requests this process is still computing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight: Dict[str, Dict[str, Any]] = {}
        self.accepted = 0
        self.completed = 0
        self.max_seen_seq = 0

    def register(self, request_id: str, seq: int) -> None:
        with self._lock:
            self._in_flight[request_id] = {"seq": seq, "started": time.perf_counter()}
            self.accepted += 1
            if seq > self.max_seen_seq:
                self.max_seen_seq = seq

    def deregister(self, request_id: str) -> None:
        with self._lock:
            if self._in_flight.pop(request_id, None) is not None:
                self.completed += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.perf_counter()
            entries = [
                InflightEntry(
                    request_id=rid,
                    seq=int(meta["seq"]),
                    elapsed_ms=round((now - float(meta["started"])) * 1000.0, 3),
                )
                for rid, meta in self._in_flight.items()
            ]
            return {
                "in_flight": len(self._in_flight),
                "accepted": self.accepted,
                "completed": self.completed,
                "in_flight_requests": entries,
                "max_seen_seq": self.max_seen_seq,
            }


class _InflightMiddleware:
    """Pure-ASGI middleware registering every inference before dispatch.

    Registration is the first thing that happens to a POST /infer request —
    before routing, validation and the threadpool — so a request the worker
    received is visible in ``/status`` (practically) immediately.  Removal
    happens when the request settles, unless the endpoint transferred
    ownership to a background computation (the ``drop_response`` demo hook),
    which removes it when the computation actually ends.
    """

    def __init__(self, app: Callable, registry: _InflightRegistry) -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/infer"
        ):
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        request_id = headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        try:
            seq = int(headers.get(SEQ_HEADER) or 0)
        except ValueError:
            seq = 0
        # Shared with the endpoint via request.scope: the drop_response path
        # flips ``transferred`` so the background thread owns deregistration.
        scope["qc_inflight"] = {"request_id": request_id, "transferred": False}
        self.registry.register(request_id, seq)
        try:
            await self.app(scope, receive, send)
        finally:
            if not scope["qc_inflight"]["transferred"]:
                self.registry.deregister(request_id)


def create_worker_app(version: str, slot_id: str, model_path: str) -> FastAPI:
    model: Model = load_model(model_path, version=version)

    app = FastAPI(
        title=f"QC classifier worker {version}",
        description="Internal per-version worker. Not intended to be exposed publicly.",
    )
    started_at = time.time()
    inflight = _InflightRegistry()
    app.add_middleware(_InflightMiddleware, registry=inflight)

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"ok": True, "version": version, "pid": os.getpid()}

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        snap = inflight.snapshot()
        return StatusResponse(
            version=version,
            slot_id=slot_id,
            pid=os.getpid(),
            in_flight=snap["in_flight"],
            accepted=snap["accepted"],
            completed=snap["completed"],
            in_flight_requests=snap["in_flight_requests"],
            max_seen_seq=snap["max_seen_seq"],
            started_at=started_at,
            uptime_seconds=round(time.time() - started_at, 3),
            n_features=model.n_features,
            n_classes=model.n_classes,
            labels=list(model.labels),
        )

    def _compute(features: np.ndarray, slow_ms: int):
        # Deliberate demo delay.  The work runs in a *thread*, so an API
        # timeout, a client disconnect or a broken response connection does
        # not abort it; the in-flight table only clears once this returns.
        if slow_ms:
            time.sleep(slow_ms / 1000.0)
        return model.predict(features)

    def _background_compute(request_id: str, features: np.ndarray, slow_ms: int) -> None:
        """Owns the in-flight entry for a drop_response request to the end."""
        try:
            _compute(features, slow_ms)
        finally:
            inflight.deregister(request_id)

    async def _broken_stream():
        # The response headers (200) have already been sent; raising here
        # aborts the connection mid-body, so the API sees a transport error
        # while the computation keeps running in the background thread.
        yield b'{"partial": true, "note": "response connection dropped by demo hook"}'
        raise RuntimeError("simulated response connection break (drop_response)")

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest, request: Request) -> Any:  # sync: threadpool
        features = np.asarray(req.features, dtype=np.float64)
        if features.ndim != 1 or features.shape[0] != model.n_features:
            raise HTTPException(
                status_code=400,
                detail=f"expected {model.n_features} features, got {features.shape[0]}",
            )
        if not np.all(np.isfinite(features)):
            raise HTTPException(status_code=400, detail="features contain NaN or infinite values")

        if req.drop_response:
            qc = request.scope.get("qc_inflight") or {}
            request_id = qc.get("request_id") or uuid.uuid4().hex
            qc["transferred"] = True  # background thread owns deregistration
            threading.Thread(
                target=_background_compute,
                args=(request_id, features, req.slow_ms),
                daemon=True,
            ).start()
            return StreamingResponse(_broken_stream(), media_type="application/json")

        inference_id = uuid.uuid4().hex
        started = time.perf_counter()
        label, score = _compute(features, req.slow_ms)
        elapsed = (time.perf_counter() - started) * 1000.0
        return InferResponse(
            version=version,
            label=label,
            score=score,
            inference_id=inference_id,
            elapsed_ms=round(elapsed, 3),
        )

    return app

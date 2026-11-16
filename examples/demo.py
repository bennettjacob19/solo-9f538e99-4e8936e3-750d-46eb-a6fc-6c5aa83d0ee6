#!/usr/bin/env python3
"""End-to-end demo of the hot-swap service, runnable against a live API.

Usage::

    python examples/demo.py [BASE_URL]

Covers the scenarios reviewers care about:

1. normal inference, then a successful hot swap;
2. a failed candidate load (old version keeps serving);
3. slow inference with a client timeout, followed by a swap and a third
   upload that is refused until the slow work finishes and the slot is freed;
4. an API-to-worker connection break mid-inference: the caller gets 502, the
   request keeps its slot as "unconfirmed", the old process is not stopped
   early, and reconciliation releases the slot once the worker confirms the
   computation ended.
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
LABELS = ["pass", "scratch", "dent"]
W1 = [[1.2, -0.4, 0.0, 0.3], [-0.2, 0.9, 0.5, -0.1], [0.1, -0.8, 1.1, 0.2]]
W2 = [[0.3, 1.1, -0.2, 0.1], [1.3, -0.3, 0.1, -0.2], [-0.4, 0.2, 0.9, 0.7]]
FEATURES = [0.7, 0.2, 0.1, 0.5]

client = httpx.Client(base_url=BASE, timeout=60.0)


def show(label, resp):
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"  {label:<46} {resp.status_code}  {json.dumps(body)[:160]}")
    return resp


def slots():
    st = client.get("/admin/status").json()
    layout = [
        (s["slot_id"], s["version"], s["process_state"],
         f"in_flight={s['in_flight']}", f"unconfirmed={s['unconfirmed']}")
        for s in st["slots"]
    ]
    print(f"  status: active={st['active_version']} slots={layout}")
    return st


def artifact(version, weights, **extra):
    return {"version": version, "weights": weights, "bias": [0.0, 0.0, 0.0], "labels": LABELS, **extra}


def wait_slots_in_use(n, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = slots()
        if st["slots_in_use"] == n:
            return st
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for slots_in_use == {n}")


def main() -> int:
    print("\n== 0. baseline load ==")
    show("upload v1", client.post("/admin/models", json=artifact("v1", W1)))
    show("infer", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 1. successful hot swap ==")
    show("upload v2", client.post("/admin/models", json=artifact("v2", W2)))
    show("infer after swap", client.post("/infer", json={"features": FEATURES}))
    slots()

    print("\n== 2. failed candidate load leaves v2 active ==")
    show(
        "upload bad-load",
        client.post(
            "/admin/models", json=artifact("bad-load", W1, simulate_load_failure=True)
        ),
    )
    show("infer (still v2)", client.post("/infer", json={"features": FEATURES}))
    show(
        "upload invalid artifact",
        client.post("/admin/models", json=artifact("broken", [[1.0, 2.0]])),
    )
    slots()

    print("\n== 3. slow inference + client timeout ==")
    time.sleep(1.0)  # let the v1 drain finish and free a slot
    show(
        "slow infer, timeout_ms=150",
        client.post("/infer", json={"features": FEATURES, "slow_ms": 6000, "timeout_ms": 150}),
    )
    st = slots()
    print(f"  in-flight work still running in model process: {st['total_in_flight']}")

    print("\n== 4. swap while the slow call is still running ==")
    show("upload v3", client.post("/admin/models", json=artifact("v3", W1)))
    slots()
    show("upload v4 (third process)", client.post("/admin/models", json=artifact("v4", W2)))
    show("infer (v3 already serving)", client.post("/infer", json={"features": FEATURES}))

    print("\n== 5. wait for the slow work to finish; slot is released ==")
    wait_slots_in_use(1)
    show("upload v4 now", client.post("/admin/models", json=artifact("v4", W2)))
    slots()

    print("\n== 6. API-to-worker connection break mid-inference ==")
    wait_slots_in_use(1)
    print("  worker accepts, drops the response connection, keeps computing 8s")
    show(
        "infer drop_response (502)",
        client.post(
            "/infer",
            json={"features": FEATURES, "slow_ms": 8000, "drop_response": True},
        ),
    )
    st = slots()
    print(f"  still occupying the slot as unconfirmed: {st['total_unconfirmed']}")
    show("infer (current version keeps serving)", client.post("/infer", json={"features": FEATURES}))

    print("\n== 7. swap uses the other slot; third version refused ==")
    show("upload v5", client.post("/admin/models", json=artifact("v5", W1)))
    slots()
    show("upload v6 (third process)", client.post("/admin/models", json=artifact("v6", W2)))

    print("\n== 8. worker finishes; reconciliation confirms and frees the slot ==")
    wait_slots_in_use(1)
    st = slots()
    print(f"  unconfirmed now: {st['total_unconfirmed']}  "
          f"(accepted={st['total_accepted']} completed={st['total_completed']})")
    show("upload v6 now", client.post("/admin/models", json=artifact("v6", W2)))
    slots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

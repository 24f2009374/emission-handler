"""
Emission Handler — deterministic checkpoint state machine.

Models a shipment moving through a directed graph of nodes (vendor,
checkpoints, site). Each edge represents a transport leg with an expected
transit time T. The source node "emits" a dispatch signal; the target node
acknowledges and waits; when the leg resolves, it emits either an on-time
or delayed signal. Downstream legs can be shifted based on upstream delays.

This file is self-contained (core logic + scenarios + HTTP handler) so it
deploys cleanly as a single Vercel Python Function with no cross-file
imports.
"""

import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Core state machine
# ---------------------------------------------------------------------------

class EdgeState(str, Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    ACK_WAITING = "ack_waiting"
    ON_TIME = "on_time"
    DELAYED = "delayed"
    UNKNOWN = "unknown"
    CONFLICTED = "conflicted"


class EmissionType(str, Enum):
    DISPATCHED = "dispatched"
    ACK_WAITING = "ack_waiting"
    ARRIVED = "arrived"
    DELAYED = "delayed"
    SILENCE = "silence"
    CONFLICT = "conflict"
    CONTEXT_ADJUST = "context_adjust"


@dataclass
class Emission:
    edge_id: str
    type: EmissionType
    at: str
    detail: str = ""

    def to_dict(self):
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class Edge:
    id: str
    source: str
    target: str
    expected_transit_minutes: int
    dispatched_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    state: EdgeState = EdgeState.PENDING
    note: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "expected_transit_minutes": self.expected_transit_minutes,
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "state": self.state.value,
            "note": self.note,
        }


class EmissionHandler:
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.edges: Dict[str, Edge] = {}
        self.log: List[Emission] = []

    def add_edge(self, edge_id, source, target, expected_transit_minutes):
        self.edges[edge_id] = Edge(
            id=edge_id, source=source, target=target,
            expected_transit_minutes=expected_transit_minutes,
        )

    def _emit(self, edge_id, etype, at, detail=""):
        e = Emission(edge_id=edge_id, type=etype, at=at.isoformat(), detail=detail)
        self.log.append(e)
        return e

    def dispatch(self, edge_id, at: datetime):
        edge = self.edges[edge_id]
        edge.dispatched_at = at
        edge.deadline = at + timedelta(minutes=edge.expected_transit_minutes)
        edge.state = EdgeState.IN_TRANSIT
        self._emit(
            edge_id, EmissionType.DISPATCHED, at,
            f"{edge.source} confirmed dispatch to {edge.target}, "
            f"deadline {edge.deadline.isoformat()}",
        )

    def acknowledge(self, edge_id, at: datetime):
        edge = self.edges[edge_id]
        edge.state = EdgeState.ACK_WAITING
        self._emit(
            edge_id, EmissionType.ACK_WAITING, at,
            f"{edge.target} acknowledged, waiting until {edge.deadline.isoformat()}",
        )

    def resolve(self, edge_id, at: datetime):
        edge = self.edges[edge_id]
        if at <= edge.deadline:
            edge.state = EdgeState.ON_TIME
            self._emit(edge_id, EmissionType.ARRIVED, at,
                        f"{edge.target} confirms on-time arrival")
        else:
            edge.state = EdgeState.DELAYED
            overrun = at - edge.deadline
            self._emit(edge_id, EmissionType.DELAYED, at,
                        f"{edge.target} confirms delayed arrival, overrun {overrun}")
        return edge.state

    def apply_context_adjustment(self, edge_id, extra_minutes, reason, at):
        edge = self.edges[edge_id]
        edge.deadline += timedelta(minutes=extra_minutes)
        self._emit(
            edge_id, EmissionType.CONTEXT_ADJUST, at,
            f"deadline shifted +{extra_minutes}m due to known cause: {reason}",
        )

    def report_silence(self, edge_id, checked_at: datetime, grace_minutes=0):
        edge = self.edges[edge_id]
        if edge.deadline and checked_at > edge.deadline + timedelta(minutes=grace_minutes):
            edge.state = EdgeState.UNKNOWN
            self._emit(
                edge_id, EmissionType.SILENCE, checked_at,
                f"No signal from {edge.target} past deadline+grace — cannot resolve "
                f"deterministically. Requires inference from historical patterns / "
                f"unstructured status text.",
            )
        return edge.state

    def report_conflict(self, edge_id, at: datetime, signal_a: str, signal_b: str):
        edge = self.edges[edge_id]
        edge.state = EdgeState.CONFLICTED
        self._emit(
            edge_id, EmissionType.CONFLICT, at,
            f"Conflicting signals for {edge.target}: '{signal_a}' vs '{signal_b}'. "
            f"Requires trust-weighted resolution, not solvable by timeout logic alone.",
        )

    def snapshot(self):
        return {
            "edges": [e.to_dict() for e in self.edges.values()],
            "log": [e.to_dict() for e in self.log],
        }


# ---------------------------------------------------------------------------
# Deterministic test scenarios
# ---------------------------------------------------------------------------

BASE = datetime(2026, 7, 7, 8, 0, 0)


def scenario_normal():
    h = EmissionHandler(nodes=["Vendor", "CheckpointB", "Site"])
    h.add_edge("A-B", "Vendor", "CheckpointB", expected_transit_minutes=120)
    h.add_edge("B-C", "CheckpointB", "Site", expected_transit_minutes=90)

    h.dispatch("A-B", BASE)
    h.acknowledge("A-B", BASE)
    h.resolve("A-B", BASE + timedelta(minutes=110))

    t2 = BASE + timedelta(minutes=110)
    h.dispatch("B-C", t2)
    h.acknowledge("B-C", t2)
    h.resolve("B-C", t2 + timedelta(minutes=80))

    return {
        "title": "Normal path — everything on time",
        "explains": "Deterministic handshake logic is sufficient here. No inference needed.",
        **h.snapshot(),
    }


def scenario_single_delay_cascade():
    h = EmissionHandler(nodes=["Vendor", "CheckpointB", "Site"])
    h.add_edge("A-B", "Vendor", "CheckpointB", expected_transit_minutes=120)
    h.add_edge("B-C", "CheckpointB", "Site", expected_transit_minutes=90)

    h.dispatch("A-B", BASE)
    h.acknowledge("A-B", BASE)
    h.resolve("A-B", BASE + timedelta(minutes=170))  # 50 min late

    t2 = BASE + timedelta(minutes=170)
    h.dispatch("B-C", t2)
    h.acknowledge("B-C", t2)
    h.resolve("B-C", t2 + timedelta(minutes=85))

    return {
        "title": "Single-leg delay cascades downstream",
        "explains": (
            "Deterministic system correctly detects the delay AND propagates the shifted "
            "start time to the next leg. This is the reactive baseline — it tells you a "
            "delay happened, after it happened."
        ),
        **h.snapshot(),
    }


def scenario_silence():
    h = EmissionHandler(nodes=["Vendor", "CheckpointB"])
    h.add_edge("A-B", "Vendor", "CheckpointB", expected_transit_minutes=120)

    h.dispatch("A-B", BASE)
    h.acknowledge("A-B", BASE)
    h.report_silence("A-B", checked_at=BASE + timedelta(minutes=180), grace_minutes=30)

    return {
        "title": "Silent node — no signal at all",
        "explains": (
            "Deterministic logic hits a wall: it can flag that the deadline passed with no "
            "confirmation, but has no way to estimate where the shipment actually is or how "
            "delayed it likely is. This is where a GNN trained on historical route/vendor "
            "behavior, or a RAG lookup over vendor emails, fills the gap instead of showing "
            "a blank state."
        ),
        **h.snapshot(),
    }


def scenario_conflicting_signals():
    h = EmissionHandler(nodes=["Vendor", "CheckpointB"])
    h.add_edge("A-B", "Vendor", "CheckpointB", expected_transit_minutes=120)

    h.dispatch("A-B", BASE)
    h.acknowledge("A-B", BASE)
    h.report_conflict(
        "A-B",
        at=BASE + timedelta(minutes=115),
        signal_a="GPS ping: still 12km from CheckpointB",
        signal_b="Manual checkpoint log: marked arrived",
    )

    return {
        "title": "Conflicting signals from two independent sources",
        "explains": (
            "Deterministic handshake logic has no built-in way to pick a winner between two "
            "disagreeing sources. This needs a trust-weighted resolution model — e.g. a "
            "learned reliability score per signal source, a natural fit for the GNN layer "
            "(source reliability as a learned feature, not a hardcoded rule)."
        ),
        **h.snapshot(),
    }


def scenario_known_context_delay():
    h = EmissionHandler(nodes=["Vendor", "CheckpointB"])
    h.add_edge("A-B", "Vendor", "CheckpointB", expected_transit_minutes=120)

    h.dispatch("A-B", BASE)
    h.apply_context_adjustment(
        "A-B", extra_minutes=180,
        reason="Port customs holiday, pre-confirmed 3-day hold",
        at=BASE + timedelta(minutes=5),
    )
    h.acknowledge("A-B", BASE + timedelta(minutes=5))
    h.resolve("A-B", BASE + timedelta(minutes=290))

    return {
        "title": "Pre-known contextual delay folded into deadline",
        "explains": (
            "Unlike the silence/conflict cases, this is context already known ahead of time "
            "(holidays, pre-confirmed customs holds). Deterministic logic handles it cleanly "
            "by adjusting the deadline itself — no inference required. This is the boundary "
            "case that shows what's rule-based vs. what genuinely needs learned prediction."
        ),
        **h.snapshot(),
    }

def scenario_long_chain_recovery():
    h = EmissionHandler(nodes=["Factory", "Port", "Customs", "RegionalWarehouse", "Site"])
    h.add_edge("F-P", "Factory", "Port", expected_transit_minutes=180)
    h.add_edge("P-C", "Port", "Customs", expected_transit_minutes=60)
    h.add_edge("C-W", "Customs", "RegionalWarehouse", expected_transit_minutes=240)
    h.add_edge("W-S", "RegionalWarehouse", "Site", expected_transit_minutes=90)

    t0 = BASE
    h.dispatch("F-P", t0)
    h.acknowledge("F-P", t0)
    t1 = t0 + timedelta(minutes=170)
    h.resolve("F-P", t1)  # on time

    h.dispatch("P-C", t1)
    h.acknowledge("P-C", t1)
    t2 = t1 + timedelta(minutes=75)
    h.resolve("P-C", t2)  # delayed by 15 min

    h.dispatch("C-W", t2)
    h.apply_context_adjustment(
        "C-W", extra_minutes=180,
        reason="Pre-confirmed customs hold", at=t2 + timedelta(minutes=5),
    )
    h.acknowledge("C-W", t2 + timedelta(minutes=5))
    t3 = t2 + timedelta(minutes=400)
    h.resolve("C-W", t3)  # on time relative to adjusted deadline

    h.dispatch("W-S", t3)
    h.acknowledge("W-S", t3)
    h.report_silence("W-S", checked_at=t3 + timedelta(minutes=140), grace_minutes=20)

    return {
        "title": "Long multi-hop chain — delay, anticipated hold, then silence",
        "explains": (
            "Four hops, three different outcomes in one run: an on-time leg, a genuine "
            "delay, a large but pre-known hold that's absorbed cleanly by a context "
            "adjustment, and a final leg that goes silent. This is closer to what a real "
            "multi-leg shipment looks like — most legs behave, but the failure modes stack."
        ),
        **h.snapshot(),
    }


def scenario_branching_shipment_split():
    h = EmissionHandler(nodes=["Vendor", "DistributionHub", "SiteA", "SiteB"])
    h.add_edge("V-H", "Vendor", "DistributionHub", expected_transit_minutes=100)
    h.add_edge("H-A", "DistributionHub", "SiteA", expected_transit_minutes=70)
    h.add_edge("H-B", "DistributionHub", "SiteB", expected_transit_minutes=70)

    t0 = BASE
    h.dispatch("V-H", t0)
    h.acknowledge("V-H", t0)
    t1 = t0 + timedelta(minutes=95)
    h.resolve("V-H", t1)  # on time

    h.dispatch("H-A", t1)
    h.acknowledge("H-A", t1)
    h.resolve("H-A", t1 + timedelta(minutes=60))  # on time

    h.dispatch("H-B", t1)
    h.acknowledge("H-B", t1)
    h.report_conflict(
        "H-B",
        at=t1 + timedelta(minutes=65),
        signal_a="Driver app: en route, ETA 15 more minutes",
        signal_b="SiteB gate log: shipment marked received",
    )

    return {
        "title": "Branching shipment — shared upstream, diverging downstream fates",
        "explains": (
            "One shipment splits at a distribution hub into two site-bound legs. Both "
            "start from an identical, on-time upstream state, but one resolves cleanly "
            "while the other hits a signal conflict. This shows the graph isn't just a "
            "line — risk doesn't distribute evenly once a shipment forks."
        ),
        **h.snapshot(),
    }


SCENARIO_TITLES = {
    "normal": "Normal path — everything on time",
    "single_delay_cascade": "Single-leg delay cascades downstream",
    "silence": "Silent node — no signal at all",
    "conflicting_signals": "Conflicting signals from two independent sources",
    "known_context_delay": "Pre-known contextual delay folded into deadline",
    "long_chain_recovery": "Long multi-hop chain — delay, anticipated hold, then silence",
    "branching_shipment_split": "Branching shipment — shared upstream, diverging downstream fates",
}

SCENARIOS = {
    "normal": scenario_normal,
    "single_delay_cascade": scenario_single_delay_cascade,
    "silence": scenario_silence,
    "conflicting_signals": scenario_conflicting_signals,
    "known_context_delay": scenario_known_context_delay,
    "long_chain_recovery": scenario_long_chain_recovery,
    "branching_shipment_split": scenario_branching_shipment_split,
}


# ---------------------------------------------------------------------------
# Vercel HTTP handler
# ---------------------------------------------------------------------------

class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        case = qs.get("case", [None])[0]

        if case is None:
            self._send_json(200, {
                "scenarios": [
                    {"id": k, "title": v} for k, v in SCENARIO_TITLES.items()
                ]
            })
            return

        if case not in SCENARIOS:
            self._send_json(400, {"error": f"unknown scenario '{case}'"})
            return

        result = SCENARIOS[case]()
        self._send_json(200, result)

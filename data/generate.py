"""
Synthetic Construction Supply Chain Data Generator
====================================================

Produces a relational SQLite database (+ optional CSV / graph-JSON exports)
modeling material orders flowing through multi-hop shipment routes, with
realistic (not uniform-random) delay structure baked in:

  - Vendor reliability is a hidden ground-truth score that biases delay
    rate/severity but is never exposed as a plain feature -- it's what a
    model would have to infer from behavior.
  - Delay magnitude is right-skewed (lognormal), not symmetric.
  - A handful of shared "shock" context events (customs holds, holidays,
    weather) hit every order passing through a node in a date window --
    this is what gives you real correlation/clustering to find in EDA,
    instead of independent per-order noise.
  - A configurable fraction of legs resolve as SILENCE or CONFLICT instead
    of a clean on_time/delayed outcome -- these map directly to the two
    failure modes the emission handler demo already illustrates.

All tunable knobs live in the CONFIG dataclass below. Nothing else in the
file should need editing to change scale/shape of the dataset.

Usage:
    python generate.py
"""

import sqlite3
import json
import csv
import random
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# HYPERPARAMETERS -- edit these to scale the dataset up/down or reshape it
# ---------------------------------------------------------------------------

@dataclass
class Config:
    SEED: int = 42

    # --- scale ---
    NUM_VENDORS: int = 20
    NUM_PROJECTS: int = 6
    NUM_ROUTE_TEMPLATES: int = 6
    NUM_ORDERS: int = 800

    # --- vendor reliability distribution ---
    # Beta(alpha, beta) skewed toward reliable vendors with a real tail of
    # unreliable ones. Higher alpha relative to beta = more reliable vendors.
    VENDOR_RELIABILITY_ALPHA: float = 6.0
    VENDOR_RELIABILITY_BETA: float = 2.0

    # --- delay behavior ---
    # Baseline chance ANY leg is delayed at all, independent of vendor.
    BASE_DELAY_PROB: float = 0.18
    # How much a vendor's (1 - reliability) adds to that baseline probability.
    RELIABILITY_DELAY_WEIGHT: float = 0.55
    # Lognormal params (in minutes) for delay MAGNITUDE when a delay occurs.
    # mean/sigma are of the underlying normal (i.e. log-space).
    DELAY_LOGNORM_MEAN: float = 3.6   # exp(3.6) ~= 37 min typical
    DELAY_LOGNORM_SIGMA: float = 0.9  # controls the long tail

    # --- failure modes (map to emission handler's silence/conflict states) ---
    SILENCE_RATE: float = 0.11      # fraction of legs that go silent instead of resolving
    CONFLICT_RATE: float = 0.06     # fraction of legs that resolve as conflicting signals

    # --- context events (shared shocks) ---
    NUM_CONTEXT_EVENTS: int = 10
    CONTEXT_PREANNOUNCED_FRACTION: float = 0.5  # rest are "unknown until they hit"
    CONTEXT_EXTRA_MINUTES_RANGE: Tuple[int, int] = (120, 4320)  # 2h to 3 days

    # --- order-level fields ---
    MATERIAL_TYPES: Tuple[str, ...] = (
        "Structural Steel", "Rebar", "Precast Concrete", "Electrical Conduit",
        "HVAC Ductwork", "Curtain Wall Glass", "Plumbing Fixtures",
        "Insulation", "Drywall", "Roofing Membrane",
    )
    QUANTITY_RANGE: Tuple[int, int] = (10, 2000)
    UNIT_COST_RANGE: Tuple[float, float] = (15.0, 850.0)
    ORDER_DATE_START: str = "2025-09-01"
    ORDER_DATE_END: str = "2026-06-01"
    APPROVAL_DELAY_DAYS_RANGE: Tuple[int, int] = (1, 14)

    # --- output ---
    OUTPUT_DIR: str = "/home/claude/synthdata/output"
    DB_FILENAME: str = "supply_chain.db"
    ALSO_EXPORT_CSV: bool = True
    ALSO_EXPORT_GRAPH_JSON: bool = True


CFG = Config()


# ---------------------------------------------------------------------------
# Route templates -- node role sequences with baseline expected transit times
# ---------------------------------------------------------------------------
# Each template is a list of (node_role, expected_transit_minutes_to_next).
# The last node has no outgoing leg, so its transit value is unused (None).

ROUTE_TEMPLATE_POOL = [
    [("Factory", 180), ("Port", 60), ("Customs", 240), ("RegionalWarehouse", 90), ("Site", None)],
    [("Factory", 90), ("DistributionHub", 45), ("Site", None)],
    [("Fabricator", 150), ("QAInspection", 30), ("RegionalWarehouse", 100), ("Site", None)],
    [("Vendor", 60), ("LocalDepot", 40), ("Site", None)],
    [("Factory", 220), ("Port", 80), ("RegionalWarehouse", 110), ("Site", None)],
    [("Fabricator", 100), ("DistributionHub", 50), ("SiteA", None)],
    [("Fabricator", 100), ("DistributionHub", 50), ("SiteB", None)],
    [("OverseasFactory", 300), ("Port", 90), ("Customs", 300), ("Port2", 60), ("RegionalWarehouse", 120), ("Site", None)],
]


def random_date(start: str, end: str, rng: random.Random) -> datetime:
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    delta_days = (d1 - d0).days
    return d0 + timedelta(days=rng.randint(0, delta_days))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE vendors (
    vendor_id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    hidden_reliability REAL  -- ground truth only; not a modeling feature
);

CREATE TABLE projects (
    project_id INTEGER PRIMARY KEY,
    name TEXT
);

CREATE TABLE route_templates (
    route_id INTEGER PRIMARY KEY,
    label TEXT
);

CREATE TABLE route_nodes (
    route_id INTEGER,
    seq_index INTEGER,
    node_role TEXT,
    expected_transit_minutes INTEGER,  -- to next node; NULL for terminal node
    PRIMARY KEY (route_id, seq_index),
    FOREIGN KEY (route_id) REFERENCES route_templates(route_id)
);

CREATE TABLE context_events (
    event_id INTEGER PRIMARY KEY,
    node_role TEXT,
    start_date TEXT,
    end_date TEXT,
    extra_minutes INTEGER,
    reason TEXT,
    pre_announced INTEGER  -- 1 = known ahead of time, 0 = unannounced shock
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    vendor_id INTEGER,
    project_id INTEGER,
    route_id INTEGER,
    material_type TEXT,
    quantity INTEGER,
    unit_cost REAL,
    order_date TEXT,
    approval_date TEXT,
    roj_date TEXT,
    final_status TEXT,  -- worst leg state across the whole order
    FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (route_id) REFERENCES route_templates(route_id)
);

CREATE TABLE legs (
    leg_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    seq_index INTEGER,
    source_role TEXT,
    target_role TEXT,
    planned_transit_minutes INTEGER,
    context_extra_minutes INTEGER,      -- 0 if no context event applied
    context_reason TEXT,
    dispatched_at TEXT,
    deadline TEXT,
    resolved_at TEXT,                   -- NULL if silence
    state TEXT,                         -- on_time | delayed | silence | conflict
    conflict_signal_a TEXT,
    conflict_signal_b TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(cfg: Config):
    rng = random.Random(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    db_path = os.path.join(cfg.OUTPUT_DIR, cfg.DB_FILENAME)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    # --- vendors ---
    vendor_categories = ["Steel", "Concrete", "Electrical", "HVAC", "Glazing", "Plumbing", "General Fabrication"]
    vendors = []
    for vid in range(1, cfg.NUM_VENDORS + 1):
        reliability = rng.betavariate(cfg.VENDOR_RELIABILITY_ALPHA, cfg.VENDOR_RELIABILITY_BETA)
        name = f"Vendor-{vid:03d}"
        category = rng.choice(vendor_categories)
        vendors.append((vid, name, category, reliability))
    cur.executemany("INSERT INTO vendors VALUES (?,?,?,?)", vendors)

    # --- projects ---
    projects = [(pid, f"Project-{pid:02d}") for pid in range(1, cfg.NUM_PROJECTS + 1)]
    cur.executemany("INSERT INTO projects VALUES (?,?)", projects)

    # --- route templates ---
    chosen_templates = rng.sample(ROUTE_TEMPLATE_POOL, k=min(cfg.NUM_ROUTE_TEMPLATES, len(ROUTE_TEMPLATE_POOL)))
    route_rows = []
    route_node_rows = []
    for rid, template in enumerate(chosen_templates, start=1):
        route_rows.append((rid, f"Route-{rid:02d}"))
        for seq_idx, (role, transit) in enumerate(template):
            route_node_rows.append((rid, seq_idx, role, transit))
    cur.executemany("INSERT INTO route_templates VALUES (?,?)", route_rows)
    cur.executemany("INSERT INTO route_nodes VALUES (?,?,?,?)", route_node_rows)

    # --- context events (shared shocks) ---
    all_roles = sorted(set(r[2] for r in route_node_rows))
    context_rows = []
    context_reasons_known = ["Public holiday closure", "Pre-confirmed customs hold", "Scheduled maintenance window"]
    context_reasons_unknown = ["Unplanned strike", "Severe weather", "Equipment breakdown", "Port congestion surge"]
    for eid in range(1, cfg.NUM_CONTEXT_EVENTS + 1):
        role = rng.choice(all_roles)
        start = random_date(cfg.ORDER_DATE_START, cfg.ORDER_DATE_END, rng)
        duration_days = rng.randint(1, 5)
        end = start + timedelta(days=duration_days)
        extra = rng.randint(*cfg.CONTEXT_EXTRA_MINUTES_RANGE)
        pre_announced = rng.random() < cfg.CONTEXT_PREANNOUNCED_FRACTION
        reason = rng.choice(context_reasons_known if pre_announced else context_reasons_unknown)
        context_rows.append((eid, role, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                              extra, reason, int(pre_announced)))
    cur.executemany("INSERT INTO context_events VALUES (?,?,?,?,?,?,?)", context_rows)

    # --- orders + legs ---
    vendor_reliability = {v[0]: v[3] for v in vendors}
    order_rows = []
    leg_rows = []
    leg_id_counter = 1

    for oid in range(1, cfg.NUM_ORDERS + 1):
        vendor_id = rng.randint(1, cfg.NUM_VENDORS)
        project_id = rng.randint(1, cfg.NUM_PROJECTS)
        route_id = rng.randint(1, len(chosen_templates))
        template = chosen_templates[route_id - 1]

        material = rng.choice(cfg.MATERIAL_TYPES)
        quantity = rng.randint(*cfg.QUANTITY_RANGE)
        unit_cost = round(rng.uniform(*cfg.UNIT_COST_RANGE), 2)

        order_date = random_date(cfg.ORDER_DATE_START, cfg.ORDER_DATE_END, rng)
        approval_date = order_date + timedelta(days=rng.randint(*cfg.APPROVAL_DELAY_DAYS_RANGE))

        reliability = vendor_reliability[vendor_id]
        delay_prob = min(0.95, cfg.BASE_DELAY_PROB + cfg.RELIABILITY_DELAY_WEIGHT * (1 - reliability))

        cursor_time = approval_date
        leg_states = []
        order_legs_buffer = []

        for seq_idx in range(len(template) - 1):
            source_role, planned_transit = template[seq_idx]
            target_role = template[seq_idx + 1][0]

            dispatched_at = cursor_time
            deadline = dispatched_at + timedelta(minutes=planned_transit)

            # apply any overlapping context event at the SOURCE role
            context_extra = 0
            context_reason = None
            context_is_unannounced = False
            for (_, c_role, c_start, c_end, c_extra, c_reason, c_pre) in context_rows:
                if c_role != source_role:
                    continue
                c_start_dt = datetime.strptime(c_start, "%Y-%m-%d")
                c_end_dt = datetime.strptime(c_end, "%Y-%m-%d")
                if c_start_dt <= dispatched_at <= c_end_dt:
                    context_extra = c_extra
                    context_reason = c_reason
                    if c_pre:
                        # pre-announced: deadline absorbs it cleanly up front
                        deadline += timedelta(minutes=c_extra)
                    else:
                        # unannounced shock: contributes to delay, not flagged ahead of time
                        context_is_unannounced = True
                    break

            # decide resolution type
            roll = rng.random()
            state = None
            resolved_at = None
            conflict_a = conflict_b = None

            if roll < cfg.SILENCE_RATE:
                state = "silence"
            elif roll < cfg.SILENCE_RATE + cfg.CONFLICT_RATE:
                state = "conflict"
                conflict_a = "GPS/tracking signal: shipment still in transit"
                conflict_b = f"{target_role} checkpoint log: marked received"
            else:
                is_delayed = rng.random() < delay_prob or context_is_unannounced
                if is_delayed:
                    delay_minutes = int(rng.lognormvariate(cfg.DELAY_LOGNORM_MEAN, cfg.DELAY_LOGNORM_SIGMA))
                    if context_is_unannounced:
                        delay_minutes += context_extra  # unannounced shock adds directly
                    resolved_at = deadline + timedelta(minutes=delay_minutes)
                    state = "delayed"
                else:
                    early_buffer = rng.randint(0, max(1, planned_transit // 4))
                    resolved_at = deadline - timedelta(minutes=early_buffer)
                    state = "on_time"

            leg_states.append(state)
            order_legs_buffer.append((
                leg_id_counter, oid, seq_idx, source_role, target_role,
                planned_transit, context_extra, context_reason,
                dispatched_at.isoformat(), deadline.isoformat(),
                resolved_at.isoformat() if resolved_at else None,
                state, conflict_a, conflict_b,
            ))
            leg_id_counter += 1

            # advance cursor for next leg; if silent/conflict, assume worst-case
            # handoff time (deadline + small buffer) so the chain can continue
            cursor_time = resolved_at if resolved_at else (deadline + timedelta(minutes=30))

        leg_rows.extend(order_legs_buffer)

        # order-level final_status = worst state across its legs
        severity = {"on_time": 0, "delayed": 1, "silence": 2, "conflict": 2}
        worst = max(leg_states, key=lambda s: severity[s]) if leg_states else "on_time"

        roj_date = order_date + timedelta(days=rng.randint(30, 120))

        order_rows.append((
            oid, vendor_id, project_id, route_id, material, quantity, unit_cost,
            order_date.strftime("%Y-%m-%d"), approval_date.strftime("%Y-%m-%d"),
            roj_date.strftime("%Y-%m-%d"), worst,
        ))

    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)", order_rows)
    cur.executemany("INSERT INTO legs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", leg_rows)

    conn.commit()

    # --- optional exports ---
    if cfg.ALSO_EXPORT_CSV:
        export_csv(conn, cfg.OUTPUT_DIR)
    if cfg.ALSO_EXPORT_GRAPH_JSON:
        export_graph_json(conn, cfg.OUTPUT_DIR)

    print(f"Done. Database written to: {db_path}")
    print(f"  vendors={len(vendors)} projects={len(projects)} routes={len(route_rows)} "
          f"orders={len(order_rows)} legs={len(leg_rows)} context_events={len(context_rows)}")

    conn.close()
    return db_path


def export_csv(conn: sqlite3.Connection, out_dir: str):
    cur = conn.cursor()
    tables = ["vendors", "projects", "route_templates", "route_nodes",
              "context_events", "orders", "legs"]
    for t in tables:
        cur.execute(f"SELECT * FROM {t}")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        path = os.path.join(out_dir, f"{t}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)


def export_graph_json(conn: sqlite3.Connection, out_dir: str):
    """
    Node-link export of a SAMPLE of orders, in the same source/target/state
    shape the deployed emission-handler graph panel already consumes --
    useful for feeding real synthetic examples into that visualization or
    into NetworkX/Gephi for a network-level chart in the deck.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT order_id, seq_index, source_role, target_role, state,
               dispatched_at, resolved_at
        FROM legs
        ORDER BY order_id, seq_index
    """)
    rows = cur.fetchall()

    orders_graph = {}
    for order_id, seq_idx, source, target, state, dispatched_at, resolved_at in rows:
        g = orders_graph.setdefault(order_id, {"nodes": set(), "edges": []})
        g["nodes"].add(source)
        g["nodes"].add(target)
        g["edges"].append({
            "source": source, "target": target, "state": state,
            "dispatched_at": dispatched_at, "resolved_at": resolved_at,
        })

    out = []
    for order_id, g in orders_graph.items():
        out.append({
            "order_id": order_id,
            "nodes": sorted(g["nodes"]),
            "edges": g["edges"],
        })

    path = os.path.join(out_dir, "orders_graph.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    generate(CFG)

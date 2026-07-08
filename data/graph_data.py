"""
Builds an aggregated network view of the supply chain dataset:
Vendor -> route checkpoints (in sequence) -> Project, edge-weighted by
order/leg volume and delay rate. Run locally, commit the resulting JSON.

Usage (from repo root or anywhere):
    python3 data/build_graph_data.py
"""
import sqlite3, json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "supply_chain.db")
OUT_PATH = os.path.join(HERE, "graph_data.json")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

vendors = {r["vendor_id"]: dict(r) for r in cur.execute("SELECT * FROM vendors")}
projects = {r["project_id"]: dict(r) for r in cur.execute("SELECT * FROM projects")}
route_nodes = list(cur.execute("SELECT * FROM route_nodes ORDER BY route_id, seq_index"))
context_events = list(cur.execute("SELECT * FROM context_events"))
orders = {r["order_id"]: dict(r) for r in cur.execute("SELECT * FROM orders")}
legs = list(cur.execute("SELECT * FROM legs"))

route_last_index = {}
for rn in route_nodes:
    rid = rn["route_id"]
    route_last_index[rid] = max(route_last_index.get(rid, 0), rn["seq_index"])

nodes = {}
edges = {}

def nid_vendor(vid): return f"vendor:{vid}"
def nid_project(pid): return f"project:{pid}"
def nid_ckpt(route_id, seq_index, role): return f"ckpt:{route_id}:{seq_index}:{role}"
def ekey(a, b): return f"{a}|{b}"

for vid, v in vendors.items():
    nodes[nid_vendor(vid)] = {
        "id": nid_vendor(vid), "type": "vendor", "label": v["name"],
        "depth": 0, "info": {"category": v["category"]}, "_legs": [],
    }

for pid, p in projects.items():
    nodes[nid_project(pid)] = {
        "id": nid_project(pid), "type": "project", "label": p["name"],
        "depth": None, "info": {}, "_legs": [],
    }

for rn in route_nodes:
    nid = nid_ckpt(rn["route_id"], rn["seq_index"], rn["node_role"])
    nodes[nid] = {
        "id": nid, "type": "checkpoint", "label": rn["node_role"],
        "depth": rn["seq_index"] + 1,
        "info": {"route_id": rn["route_id"], "seq_index": rn["seq_index"], "role": rn["node_role"]},
        "_legs": [],
    }

max_ckpt_depth = max((n["depth"] for n in nodes.values() if n["type"] == "checkpoint"), default=0)
for pid in projects:
    nodes[nid_project(pid)]["depth"] = max_ckpt_depth + 1

for leg in legs:
    order = orders[leg["order_id"]]
    route_id = order["route_id"]
    seq = leg["seq_index"]
    src_node = nid_ckpt(route_id, seq, leg["source_role"])
    tgt_node = nid_ckpt(route_id, seq + 1, leg["target_role"])
    if src_node in nodes:
        nodes[src_node]["_legs"].append(leg)

    def bump(a, b, state):
        e = edges.setdefault(ekey(a, b), {"source": a, "target": b, "count": 0, "delayed": 0, "bad": 0})
        e["count"] += 1
        if state == "delayed": e["delayed"] += 1
        if state in ("silence", "conflict"): e["bad"] += 1

    if seq == 0:
        bump(nid_vendor(order["vendor_id"]), src_node, leg["state"])
    bump(src_node, tgt_node, leg["state"])
    if seq == route_last_index.get(route_id, 0):
        bump(tgt_node, nid_project(order["project_id"]), leg["state"])

for vid, v in vendors.items():
    n = nodes[nid_vendor(vid)]
    vendor_orders = [o for o in orders.values() if o["vendor_id"] == vid]
    all_legs = [l for l in legs if orders[l["order_id"]]["vendor_id"] == vid]
    resolved = [l for l in all_legs if l["state"] in ("on_time", "delayed")]
    on_time_rate = round(100 * sum(1 for l in resolved if l["state"] == "on_time") / len(resolved), 1) if resolved else None
    n["info"].update({
        "orders_count": len(vendor_orders),
        "observed_on_time_rate_pct": on_time_rate,
        "sample_orders": [
            {"order_id": o["order_id"], "material_type": o["material_type"],
             "quantity": o["quantity"], "final_status": o["final_status"]}
            for o in sorted(vendor_orders, key=lambda x: x["order_id"])[:6]
        ],
    })
    del n["_legs"]

for pid, p in projects.items():
    n = nodes[nid_project(pid)]
    project_orders = [o for o in orders.values() if o["project_id"] == pid]
    n["info"].update({
        "orders_count": len(project_orders),
        "sample_orders": [
            {"order_id": o["order_id"], "material_type": o["material_type"],
             "vendor_id": o["vendor_id"], "final_status": o["final_status"]}
            for o in sorted(project_orders, key=lambda x: x["order_id"])[:6]
        ],
    })
    del n["_legs"]

for nid, n in list(nodes.items()):
    if n["type"] != "checkpoint":
        continue
    counts = Counter(l["state"] for l in n["_legs"])
    role = n["info"]["role"]
    matching_events = [
        {"reason": e["reason"], "start_date": e["start_date"], "end_date": e["end_date"],
         "pre_announced": bool(e["pre_announced"])}
        for e in context_events if e["node_role"] == role
    ]
    n["info"].update({
        "legs_through": len(n["_legs"]),
        "state_breakdown": dict(counts),
        "context_events_for_this_role": matching_events,
    })
    del n["_legs"]

output = {"nodes": list(nodes.values()), "edges": list(edges.values())}
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {OUT_PATH}: {len(output['nodes'])} nodes, {len(output['edges'])} edges")
#!/usr/bin/env python3
"""
pannaraja_node_agent.py
Election-Defense Subsystem (v4.1.1) — Single-node runnable demo.

Features:
- Local SilaGate stub (H-index)
- Periodic federation sync with peers (HTTP POST/GET)
- Simple consensus (average Z + quorum)
- Time-lock issuance (local store)
- Dry-run safe mode: logs instead of broadcasting to X/IPFS
"""

import os
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
from aiohttp import web, ClientSession, ClientTimeout
import random
import logging

# Optional: load env from .env for dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NODE_ID = os.getenv("NODE_ID", "TH-SAFEHOUSE-01")
PEERS = json.loads(os.getenv("PEERS_JSON", "[]"))  # e.g. '[{"id":"SG-NODE-02","url":"http://203.0.113.50:4002"}]'
# Consensus config
QUORUM = float(os.getenv("QUORUM", "0.6"))  # fraction of nodes required
Z_THRESHOLD = float(os.getenv("Z_THRESHOLD", "0.75"))
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "30"))  # seconds
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")
X_BEARER = os.getenv("X_BEARER", "")  # keep empty in dev

# Simple in-memory stores
time_locks: List[Dict[str, Any]] = []
evidence_log: List[Dict[str, Any]] = []

# ---------------------------
# SilaGate stub (from your design)
# ---------------------------
class SilaGate:
    def __init__(self):
        self.H = 0.0
        self.evidence_log = []

    def ingest(self, source: str, cred: float, sev: float):
        self.H += float(cred) * float(sev)
        entry = {"source": source, "cred": cred, "sev": sev, "ts": datetime.utcnow().isoformat()}
        self.evidence_log.append(entry)
        logging.info(f"[SilaGate] Ingested: {entry} => H={self.H:.3f}")
        return self.H

sila = SilaGate()

# ---------------------------
# Utility functions
# ---------------------------
def make_intel_payload(z_tempo: float) -> Dict[str, Any]:
    return {
        "node_id": NODE_ID,
        "H_index": round(sila.H, 4),
        "Z_tempo": float(z_tempo),
        "timestamp": datetime.utcnow().isoformat()
    }

def issue_time_locked_constraint(condition_hash: str, protocol: Dict[str, Any], trigger_date: datetime):
    lock = {
        "hash": hashlib.sha256(condition_hash.encode()).hexdigest(),
        "protocol": protocol,
        "trigger": trigger_date.isoformat(),
        "status": "ARMED",
        "issued_by": NODE_ID,
        "issued_at": datetime.utcnow().isoformat()
    }
    time_locks.append(lock)
    logging.info(f"[TimeLock] ARMED: {lock['hash'][:16]}.. trigger={lock['trigger']}")
    return lock

async def dry_post_log(url: str, payload: Dict[str, Any]):
    """If dry-run: log. Else: POST."""
    logging.info(f"[POST] {url} -> {json.dumps(payload)}")
    if DRY_RUN:
        return {"status": "dry-run"}
    async with ClientSession(timeout=ClientTimeout(total=10)) as sess:
        async with sess.post(url, json=payload) as resp:
            return {"status": resp.status, "text": await resp.text()}

# ---------------------------
# Live Z generator (simulate or replace with real)
# ---------------------------
async def get_live_z():
    """
    Replace this with your real-time Z calculation.
    For demo: return a smoothed random walk that can cross the threshold.
    """
    base = 0.65 + (sila.H % 10) * 0.001  # tiny H influence
    noise = (random.random() - 0.5) * 0.12
    z = max(0.0, min(1.0, base + noise))
    return round(z, 4)

# ---------------------------
# Federation sync (push own intel; pull peers)
# ---------------------------
async def push_intel_to_peer(session: ClientSession, peer_url: str, payload: Dict[str, Any]):
    try:
        async with session.post(f"{peer_url.rstrip('/')}/sync", json=payload, timeout=10) as resp:
            text = await resp.text()
            logging.debug(f"[Sync] {peer_url} => {resp.status} {text[:200]}")
            return resp.status, text
    except Exception as e:
        logging.warning(f"[Sync] Failed to push to {peer_url}: {e}")
        return None, str(e)

async def gather_peer_intel(session: ClientSession, peer_url: str):
    try:
        async with session.get(f"{peer_url.rstrip('/')}/intel", timeout=10) as resp:
            data = await resp.json()
            return data
    except Exception as e:
        logging.warning(f"[Sync] Failed to fetch intel from {peer_url}: {e}")
        return None

async def federation_cycle(z_value: float):
    payload = make_intel_payload(z_value)
    # push to peers concurrently and collect their /intel
    async with ClientSession(timeout=ClientTimeout(total=10)) as sess:
        tasks = [push_intel_to_peer(sess, p["url"], payload) for p in PEERS]
        await asyncio.gather(*tasks, return_exceptions=True)

        # fetch peer intel
        fetch_tasks = [gather_peer_intel(sess, p["url"]) for p in PEERS]
        peer_results = await asyncio.gather(*fetch_tasks)
        # include self
        all_intel = [payload] + [r for r in peer_results if r]
        return all_intel

# ---------------------------
# Consensus logic
# ---------------------------
def consensus_check(all_intel: List[Dict[str, Any]]):
    if not all_intel:
        return False, 0.0, 0
    z_values = [float(i["Z_tempo"]) for i in all_intel if "Z_tempo" in i]
    avg_z = sum(z_values) / len(z_values)
    n_ok = sum(1 for z in z_values if z >= Z_THRESHOLD)
    quorum = len(z_values) and (n_ok / len(z_values) >= QUORUM)
    logging.info(f"[Consensus] nodes={len(z_values)} avg_z={avg_z:.3f} n_ok={n_ok} quorum_ok={quorum}")
    return quorum and avg_z > Z_THRESHOLD, avg_z, n_ok

# ---------------------------
# HTTP server (peer endpoints)
# ---------------------------
routes = web.RouteTableDef()
_local_intel_store: Dict[str, Any] = {}

@routes.post("/sync")
async def sync_handler(request):
    data = await request.json()
    # store latest intel from peer
    node = data.get("node_id", "unknown")
    _local_intel_store[node] = data
    logging.info(f"[HTTP] /sync from {node}: Z={data.get('Z_tempo')} H={data.get('H_index')}")
    return web.json_response({"status": "ok"})

@routes.get("/intel")
async def intel_handler(request):
    # Return our last broadcast intel + time_locks
    last_z = _local_intel_store.get(NODE_ID, {}).get("Z_tempo")
    payload = make_intel_payload(last_z if last_z is not None else 0.0)
    payload["time_locks"] = time_locks  # include local locks for transparency
    return web.json_response(payload)

# ---------------------------
# Main vigilance loop
# ---------------------------
async def vigilance_loop():
    # start HTTP server
    runner = web.AppRunner(web.Application(routes=routes))
    await runner.setup()
    port = int(os.getenv("PORT", "4001"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"[Node] {NODE_ID} HTTP server started on port {port}")

    try:
        while True:
            z = await get_live_z()
            logging.info(f"[Vigilance] Live Z={z:.3f}")

            # ingest simulated OSINT event sometimes
            if random.random() < 0.12:
                sila.ingest("Simulated OSINT: media spike", cred=0.7, sev=random.uniform(4.0, 9.0))

            # push/pull with peers and compute consensus
            all_intel = await federation_cycle(z)
            triggered, avg_z, n_ok = consensus_check(all_intel)

            if triggered:
                # compose protocol and issue lock
                protocol = {
                    "action": "PUBLISH_AUDIT",
                    "evidence_summary": f"AvgZ={avg_z:.3f}, nodes={len(all_intel)}",
                    "verifier": "Paññā-Rāja Cluster"
                }
                trigger_date = datetime.utcnow()  # immediate for demo; in prod set future
                lock = issue_time_locked_constraint("ELECTION_Z_ANOMALY", protocol, trigger_date)

                # optional autopublish (dry-run safe)
                audit_blob = {"title": "Audit", "z": avg_z, "time_lock": lock}
                if DRY_RUN:
                    logging.info(f"[AUTOPUBLISH dry-run] {json.dumps(audit_blob)}")
                else:
                    # Example: send to IPFS / X — implement production-safe calls here
                    logging.info("[AUTOPUBLISH] Would publish to IPFS and post to X now (not in DRY_RUN).")

            await asyncio.sleep(SYNC_INTERVAL)
    finally:
        await runner.cleanup()

# ---------------------------
# CLI / Demo entrypoint
# ---------------------------
if __name__ == "__main__":
    import signal
    loop = asyncio.get_event_loop()

    # handle Ctrl+C
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(loop.shutdown_asyncgens()))

    logging.info("Starting Paññā-Rāja Node Agent (demo). DRY_RUN=%s", DRY_RUN)
    try:
        loop.run_until_complete(vigilance_loop())
    except Exception as e:
        logging.exception("Node agent terminated: %s", e)

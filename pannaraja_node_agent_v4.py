#!/usr/bin/env python3
"""
pannaraja_node_agent_v4.py
Election-Defense Subsystem (v4.1.1) — Integrated with SilaGatiEngine (Māra/Paññā)
+ IPFS and X (Twitter) adapters with rate-limiting & retry/backoff.

Safe-by-default: DRY_RUN=true unless explicitly disabled.

Requires (example):
    pip install aiohttp python-dotenv torch torchvision ipfshttpclient requests

ENV variables (examples):
    NODE_ID            (default: TH-SAFEHOUSE-01)
    PORT               (default: 4001)
    PEERS_JSON         e.g. '[{"id":"SG-NODE-02","url":"http://127.0.0.1:4002"}]'
    QUORUM             (default: 0.6)
    Z_THRESHOLD        (default: 0.75)
    SYNC_INTERVAL      (default: 30)
    DRY_RUN            (default: true)
    IPFS_GATEWAY       (default: http://127.0.0.1:5001)
    X_BEARER           (Twitter bearer token; keep empty in dev)
    IPFS_RATE_LIMIT    (requests per minute, default 6)
    X_RATE_LIMIT       (requests per minute, default 60)
"""

import os
import asyncio
import json
import hashlib
import logging
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from aiohttp import web, ClientSession, ClientTimeout
from concurrent.futures import ThreadPoolExecutor

# Torch modules for SilaGatiEngine
try:
    import torch
    import torch.nn as nn
    import numpy as np
except Exception as e:
    torch = None
    nn = None
    np = None

# External blocking libs
import requests
try:
    import ipfshttpclient
except Exception:
    ipfshttpclient = None

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------- config (env) ----------
NODE_ID = os.getenv("NODE_ID", "TH-SAFEHOUSE-01")
PORT = int(os.getenv("PORT", "4001"))
PEERS = json.loads(os.getenv("PEERS_JSON", "[]"))
QUORUM = float(os.getenv("QUORUM", "0.6"))
Z_THRESHOLD = float(os.getenv("Z_THRESHOLD", "0.75"))
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "30"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")
IPFS_GATEWAY = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:5001")
X_BEARER = os.getenv("X_BEARER", "")
IPFS_RATE_LIMIT = int(os.getenv("IPFS_RATE_LIMIT", "6"))  # per minute
X_RATE_LIMIT = int(os.getenv("X_RATE_LIMIT", "60"))     # per minute

# concurrency / executors
thread_pool = ThreadPoolExecutor(max_workers=4)

# In-memory stores
time_locks: List[Dict[str, Any]] = []
_local_intel_store: Dict[str, Any] = {}
sila_evidence_log: List[Dict[str, Any]] = []

# ---------- Rate limiting helpers ----------
class RateLimiter:
    """Simple token-bucket style rate limiter (async)"""
    def __init__(self, rate_per_minute: int):
        self.capacity = max(1, rate_per_minute)
        self._tokens = self.capacity
        self._last = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last
            # refill tokens
            refill = elapsed * (self.capacity / 60.0)
            if refill >= 1:
                self._tokens = min(self.capacity, self._tokens + refill)
                self._last = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            # not enough tokens; compute sleep time
            need = 1 - self._tokens
            wait = math.ceil((need * 60.0) / self.capacity)
        await asyncio.sleep(wait)
        return await self.acquire()

ipfs_rl = RateLimiter(IPFS_RATE_LIMIT)
x_rl = RateLimiter(X_RATE_LIMIT)

# ---------- Sīla Gate + Samādhi + GATN (Mara/Panna) ----------
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

# If torch exists, define Mara/Panna. Else fallback to simple numpy stubs.
if torch and nn:
    class MaraGenerator(nn.Module):
        def __init__(self, latent_dim=128, seq_len=24):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, seq_len * 12),
                nn.Tanh()
            )
            self.seq_len = seq_len

        def forward(self, z):
            out = self.net(z)
            return out.view(-1, self.seq_len, 12)

    class PannaDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(24 * 12, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.Sigmoid()
            )

        def forward(self, threat_seq):
            flat = threat_seq.view(threat_seq.size(0), -1)
            constraint = self.net(flat)
            return constraint  # [0,1] vector

    # Minimal SilaGatiEngine wiring
    class SilaGatiEngine:
        def __init__(self, device="cpu"):
            self.device = device
            self.mara = MaraGenerator().to(device)
            self.panna = PannaDiscriminator().to(device)
            self.sila_gate = SilaGate()
            # fusion is a small helper
        def generate_threat(self, batch=1):
            z = torch.randn(batch, 128, device=self.device)
            with torch.no_grad():
                return self.mara(z)  # shape [batch, seq_len, 12]
        def evaluate_threat(self, threat_seq):
            with torch.no_grad():
                c = self.panna(threat_seq.to(self.device))
            return c  # tensor
else:
    # fallback simple random stubs
    class SilaGatiEngine:
        def __init__(self):
            self.sila_gate = SilaGate()
        def generate_threat(self, batch=1):
            # np-based random shape [batch, 24, 12]
            return np.random.uniform(-1, 1, size=(batch, 24, 12)).astype(float)
        def evaluate_threat(self, threat_seq):
            # produce constraint vector in [0,1]
            b = np.clip((np.abs(threat_seq).mean(axis=(1,2)) / 0.5), 0, 1)
            return b.reshape(-1, 1)  # shape [batch, 1]

# instantiate engine (device auto when torch available)
ENGINE = SilaGatiEngine() if not (torch and nn) else SilaGatiEngine(device="cpu")

# ---------- helper: compute Z from models + H ----------
def compute_z_from_models(sample_count=4) -> float:
    """
    Runs a few generator samples, evaluates with the discriminator, fuses scores,
    and returns a Z in [0,1]. H-index nudges Z upwards when high.
    """
    try:
        # generate sample threats
        threat_seq = ENGINE.generate_threat(batch=sample_count)
        # evaluate
        if torch and nn and isinstance(threat_seq, torch.Tensor):
            constraints = ENGINE.evaluate_threat(torch.tensor(threat_seq) if not isinstance(threat_seq, torch.Tensor) else threat_seq)
            # convert to scalar anomaly score: mean(1 - constraint) -> higher when discriminator sees deception
            if isinstance(constraints, torch.Tensor):
                score_vec = 1.0 - constraints.mean(dim=1)  # per-sample anomaly
                z = float(score_vec.mean().cpu().numpy())
        else:
            # numpy fallback
            constraints = ENGINE.evaluate_threat(threat_seq)  # e.g., shape [batch, 1]
            anomaly = 1.0 - np.array(constraints).reshape(-1)
            z = float(anomaly.mean().item() if hasattr(anomaly.mean(), "item") else anomaly.mean())
    except Exception as e:
        logging.exception("Model-based Z computation failed; falling back to random.")
        z = random.random() * 0.6 + 0.2

    # nudge with H influence (normalized): H_scaled = tanh(H/20)
    H_scaled = math.tanh(ENGINE.sila_gate.H / 20.0)
    combined = z * 0.7 + H_scaled * 0.3
    combined = max(0.0, min(1.0, combined))
    return round(combined, 4)

# ---------- IPFS Adapter (blocking ipfshttpclient in thread) ----------
async def publish_to_ipfs(obj: Dict[str, Any], retry=3) -> Optional[str]:
    await ipfs_rl.acquire()
    if DRY_RUN:
        logging.info(f"[IPFS dry-run] Would publish: {json.dumps(obj)[:500]}")
        return None
    if not ipfshttpclient:
        logging.error("ipfshttpclient missing. Install ipfshttpclient or set DRY_RUN=true.")
        return None
    last_exc = None
    for attempt in range(1, retry + 1):
        try:
            # run in thread because ipfshttpclient is sync
            loop = asyncio.get_event_loop()
            cid = await loop.run_in_executor(thread_pool, _ipfs_add_json, obj)
            logging.info(f"[IPFS] Published: cid={cid}")
            return cid
        except Exception as e:
            logging.warning(f"[IPFS] attempt {attempt} failed: {e}")
            last_exc = e
            await asyncio.sleep(2 ** attempt)
    logging.error(f"[IPFS] failed after {retry} attempts: {last_exc}")
    return None

def _ipfs_add_json(obj: Dict[str, Any]) -> str:
    # helper run in thread
    client = ipfshttpclient.connect(IPFS_GATEWAY)
    # ipfshttpclient.add_json returns CID or may require client.add_bytes
    cid = client.add_json(obj)
    client.close()
    return cid

# ---------- X/Twitter Adapter (requests + backoff) ----------
async def post_to_x(tweet: Dict[str, Any], retry=3) -> Optional[Dict[str, Any]]:
    await x_rl.acquire()
    if DRY_RUN:
        logging.info(f"[X dry-run] Would post tweet: {tweet.get('text')[:300]}")
        return {"status": "dry-run"}
    if not X_BEARER:
        logging.error("X_BEARER not configured. Enable or keep DRY_RUN.")
        return None
    headers = {"Authorization": f"Bearer {X_BEARER}", "Content-Type": "application/json"}
    url = "https://api.twitter.com/2/tweets"
    for attempt in range(1, retry + 1):
        try:
            resp = await asyncio.get_event_loop().run_in_executor(thread_pool,
                                                                  lambda: requests.post(url, json=tweet, headers=headers, timeout=15))
            if resp.status_code in (200, 201):
                logging.info(f"[X] Posted successfully: status={resp.status_code}")
                return {"status": resp.status_code, "response": resp.json()}
            if resp.status_code == 429:
                # rate-limited: inspect headers for retry-after
                wait = int(resp.headers.get("Retry-After", 60))
                logging.warning(f"[X] Rate limited. Waiting {wait}s.")
                await asyncio.sleep(wait)
                continue
            logging.warning(f"[X] unexpected status {resp.status_code}: {resp.text[:300]}")
            return {"status": resp.status_code, "text": resp.text}
        except Exception as e:
            logging.warning(f"[X] attempt {attempt} failed: {e}")
            await asyncio.sleep(2 ** attempt)
    logging.error("[X] posting failed after retries.")
    return None

# ---------- federation & consensus logic (same pattern) ----------
def make_intel_payload(z_tempo: float) -> Dict[str, Any]:
    return {
        "node_id": NODE_ID,
        "H_index": round(ENGINE.sila_gate.H, 4),
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

# network helpers
async def push_intel_to_peer(session: ClientSession, peer_url: str, payload: Dict[str, Any]):
    try:
        async with session.post(f"{peer_url.rstrip('/')}/sync", json=payload, timeout=ClientTimeout(total=10)) as resp:
            text = await resp.text()
            logging.debug(f"[Sync] {peer_url} => {resp.status} {text[:200]}")
            return resp.status, text
    except Exception as e:
        logging.warning(f"[Sync] Failed to push to {peer_url}: {e}")
        return None, str(e)

async def gather_peer_intel(session: ClientSession, peer_url: str):
    try:
        async with session.get(f"{peer_url.rstrip('/')}/intel", timeout=ClientTimeout(total=10)) as resp:
            data = await resp.json()
            return data
    except Exception as e:
        logging.warning(f"[Sync] Failed to fetch intel from {peer_url}: {e}")
        return None

async def federation_cycle(z_value: float):
    payload = make_intel_payload(z_value)
    async with ClientSession(timeout=ClientTimeout(total=10)) as sess:
        tasks = [push_intel_to_peer(sess, p["url"], payload) for p in PEERS]
        await asyncio.gather(*tasks, return_exceptions=True)
        fetch_tasks = [gather_peer_intel(sess, p["url"]) for p in PEERS]
        peer_results = await asyncio.gather(*fetch_tasks)
        all_intel = [payload] + [r for r in peer_results if r]
        return all_intel

def consensus_check(all_intel: List[Dict[str, Any]]):
    if not all_intel:
        return False, 0.0, 0
    z_values = [float(i["Z_tempo"]) for i in all_intel if "Z_tempo" in i]
    avg_z = sum(z_values) / len(z_values)
    n_ok = sum(1 for z in z_values if z >= Z_THRESHOLD)
    quorum = len(z_values) and (n_ok / len(z_values) >= QUORUM)
    logging.info(f"[Consensus] nodes={len(z_values)} avg_z={avg_z:.3f} n_ok={n_ok} quorum_ok={quorum}")
    return quorum and avg_z > Z_THRESHOLD, avg_z, n_ok

# ---------- HTTP server endpoints ----------
routes = web.RouteTableDef()

@routes.post("/sync")
async def sync_handler(request):
    data = await request.json()
    node = data.get("node_id", "unknown")
    _local_intel_store[node] = data
    logging.info(f"[HTTP] /sync from {node}: Z={data.get('Z_tempo')} H={data.get('H_index')}")
    return web.json_response({"status": "ok"})

@routes.get("/intel")
async def intel_handler(request):
    last_z = _local_intel_store.get(NODE_ID, {}).get("Z_tempo")
    payload = make_intel_payload(last_z if last_z is not None else 0.0)
    payload["time_locks"] = time_locks
    return web.json_response(payload)

# ---------- Vigilance main loop ----------
async def vigilance_loop():
    runner = web.AppRunner(web.Application(routes=routes))
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logging.info(f"[Node] {NODE_ID} HTTP server started on port {PORT} (DRY_RUN={DRY_RUN})")

    try:
        while True:
            # compute Z using model + H
            z = compute_z_from_models(sample_count=4)
            logging.info(f"[Vigilance] Live Z={z:.3f} (H={ENGINE.sila_gate.H:.3f})")

            # occasionally ingest simulated OSINT for demo
            if random.random() < 0.12:
                ENGINE.sila_gate.ingest("Simulated OSINT: media spike", cred=0.7, sev=random.uniform(4.0, 9.0))

            all_intel = await federation_cycle(z)
            triggered, avg_z, n_ok = consensus_check(all_intel)

            if triggered:
                protocol = {
                    "action": "PUBLISH_AUDIT",
                    "evidence_summary": f"AvgZ={avg_z:.3f}, nodes={len(all_intel)}",
                    "verifier": "Paññā-Rāja Cluster"
                }
                trigger_date = datetime.utcnow()  # demo immediate trigger
                lock = issue_time_locked_constraint("ELECTION_Z_ANOMALY", protocol, trigger_date)

                # build audit blob
                audit_blob = {
                    "title": "Paññā-Rāja Audit",
                    "z_score": avg_z,
                    "time_lock": lock,
                    "evidence": ENGINE.sila_gate.evidence_log[-10:]  # latest evidence
                }

                # publish to IPFS
                cid = await publish_to_ipfs(audit_blob)
                if cid:
                    audit_uri = f"ipfs.io/ipfs/{cid}"
                else:
                    audit_uri = None

                # compose tweet
                text = f"🚨 Paññā-Rāja Alert: Election anomaly detected (Z={avg_z:.3f}). Evidence: {audit_uri or '[DRY_RUN]'} #FreeMyanmar"
                tweet = {"text": text}

                res = await post_to_x(tweet)
                logging.info(f"[AUTOPUBLISH] IPFS cid={cid} tweet_result={res}")

            await asyncio.sleep(SYNC_INTERVAL)
    finally:
        await runner.cleanup()

# ---------- CLI ----------
if __name__ == "__main__":
    import signal
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(loop.shutdown_asyncgens()))

    logging.info("Starting Paññā-Rāja Node Agent v4 (model-integrated). DRY_RUN=%s", DRY_RUN)
    try:
        loop.run_until_complete(vigilance_loop())
    except Exception as e:
        logging.exception("Node agent terminated: %s", e)

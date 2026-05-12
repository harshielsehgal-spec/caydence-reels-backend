"""
rubric_persistence.py — Supabase-backed durability for rubric + creator keypoints.

The in-memory dicts in rubric_router.py wipe on every Render restart, which
kills any rubric that took 30+ seconds to compute. This module mirrors every
write to a Supabase `rubrics` table and lazy-loads on misses, so the engine
survives cold starts.

Schema (already created):
    rubric_json       jsonb
    creator_kps_json  jsonb
    duration_ms       integer
    created_at        timestamp with time zone
    updated_at        timestamp with time zone
    reel_id           text  (primary key)

Public API:
    save_rubric(reel_id, rubric, creator_kps, duration_ms)
    load_rubric(reel_id) -> (rubric, creator_kps, duration_ms) | None
    delete_rubric(reel_id) -> bool
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from rubric_schema import Rubric

# ──────────────────────────────────────────────────────────────────────────────
# Supabase client (lazy init so module imports never fail)
# ──────────────────────────────────────────────────────────────────────────────

_client = None
_init_attempted = False


def _get_client():
    """Lazily create the Supabase client. Returns None if env vars are missing."""
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("[rubric_persistence] SUPABASE_URL/SUPABASE_KEY not set — running in-memory only")
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        print("[rubric_persistence] Supabase client initialised")
    except ImportError:
        print("[rubric_persistence] supabase package not installed — pip install supabase")
        _client = None
    except Exception as e:
        print(f"[rubric_persistence] failed to init Supabase: {e}")
        _client = None
    return _client


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _kps_to_json(kps: np.ndarray) -> list:
    """Convert (N, 33, 3) numpy array → JSON-serialisable nested list."""
    return kps.astype(float).tolist()


def _kps_from_json(data: list) -> np.ndarray:
    """Convert JSON list back → (N, 33, 3) numpy array."""
    return np.array(data, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_rubric(
    reel_id: str,
    rubric: Rubric,
    creator_kps: np.ndarray,
    duration_ms: int,
) -> bool:
    """
    Upsert the rubric + creator keypoints to Supabase.
    Returns True on success, False on failure. Never raises.
    """
    client = _get_client()
    if client is None:
        return False

    try:
        payload = {
            "reel_id": reel_id,
            "rubric_json": rubric.model_dump(),
            "creator_kps_json": _kps_to_json(creator_kps),
            "duration_ms": int(duration_ms),
        }
        client.table("rubrics").upsert(payload, on_conflict="reel_id").execute()
        print(f"[rubric_persistence] saved rubric for reel '{reel_id}'")
        return True
    except Exception as e:
        print(f"[rubric_persistence] save failed for '{reel_id}': {e}")
        return False


def load_rubric(reel_id: str) -> Optional[tuple[Rubric, np.ndarray, int]]:
    """
    Fetch rubric from Supabase. Returns (rubric, creator_kps, duration_ms) or None.
    Never raises.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        resp = client.table("rubrics").select("*").eq("reel_id", reel_id).limit(1).execute()
        rows = resp.data or []
        if not rows:
            return None
        row = rows[0]
        rubric = Rubric(**row["rubric_json"])
        creator_kps = _kps_from_json(row["creator_kps_json"])
        duration_ms = int(row["duration_ms"])
        print(f"[rubric_persistence] loaded rubric for reel '{reel_id}'")
        return rubric, creator_kps, duration_ms
    except Exception as e:
        print(f"[rubric_persistence] load failed for '{reel_id}': {e}")
        return None


def delete_rubric(reel_id: str) -> bool:
    """Delete a rubric row. Returns True on success."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("rubrics").delete().eq("reel_id", reel_id).execute()
        return True
    except Exception as e:
        print(f"[rubric_persistence] delete failed for '{reel_id}': {e}")
        return False
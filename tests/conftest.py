"""Repo-wide pytest fixtures.

Security guard (autouse): no unit test may resolve a REAL credential from the
on-disk OpenClaw auth store / macOS keychain. The one-time key exposure happened
when an under-mocked auth test fell through to a host-agent store and resolved a live key.
Defaulting auth._find_auth_file to None makes any un-mocked resolution return None
instead of touching real credentials; tests that need a store patch _find_auth_file
themselves (that patch nests over this and wins).

Tests marked `smoke` are EXEMPT — the end-to-end ingestion smoke (RUN_INGEST_SMOKE=1)
legitimately exercises real auth resolution.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_real_auth_store(request, monkeypatch):
    if request.node.get_closest_marker("smoke"):
        return
    # Only neutralize the auth-FILE finder. _find_openclaw_config is left real so the
    # keyRef tests can resolve their tmp fixture config via the walk from a tmp auth
    # file; with _find_auth_file → None, an un-mocked path returns before ever calling
    # the resolver, so no real machine config/keychain is ever reached.
    monkeypatch.setattr("openclaw_brain.auth._find_auth_file", lambda: None, raising=False)


# ── Live-graph gate (2026-07-12 Insight-massacre incident) ──
# Live-fixture tests connect to the CONFIGURED Neo4j — i.e., the PRODUCTION home graph when it
# is reachable. One over-broad teardown (`insight_id STARTS WITH 'insight_'`) silently deleted
# 1,344 production Insight nodes across routine full-suite runs (zero journal — raw driver).
# Live tests are therefore OPT-IN: set RUN_LIVE_GRAPH_TESTS=1 to run them. Mocked-driver tests
# in the same files are unaffected.
import os as _os
import pytest as _pytest


def require_live_graph() -> None:
    if _os.environ.get("RUN_LIVE_GRAPH_TESTS") != "1":
        _pytest.skip(
            "live-graph test gated: set RUN_LIVE_GRAPH_TESTS=1 "
            "(writes to the CONFIGURED Neo4j — production when reachable)"
        )

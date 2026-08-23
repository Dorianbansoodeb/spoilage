"""Run one probe (or a freeform prompt) through a backend and the gate."""

from __future__ import annotations

import time

from spoilage.behavior.backends import get_backend
from spoilage.behavior.contaminate import contaminate_system
from spoilage.behavior.detect import get_gate, reasons
from spoilage.behavior.probes import Probe, require_probe
from spoilage.behavior.signals import measure_all


def _system_for(probe: Probe | None, system: str, contaminate: bool, variant: str) -> str:
    base = system
    if probe and probe.facts:
        fact_block = "Facts:\n" + "\n".join(f"- {f}" for f in probe.facts)
        base = f"{base}\n\n{fact_block}"
    if not contaminate:
        return base
    return contaminate_system(base, variant=variant, probe=probe)


def analyze_transcript(
    *,
    probe: Probe | None = None,
    probe_id: str | None = None,
    system: str | None = None,
    user: str | None = None,
    contaminate: bool = False,
    backend: str = "organism-a",
) -> dict:
    t0 = time.perf_counter()
    if probe is None and probe_id:
        probe = require_probe(probe_id)
    if probe is None and (not system or not user):
        raise ValueError("provide a probeId or both system and user")

    variant = "B" if str(backend).lower() in {"organism-b", "b"} else "A"
    sys_text = _system_for(probe, system or (probe.system if probe else ""), contaminate, variant)
    user_text = user or (probe.user if probe else "")
    client = get_backend(backend)

    if probe and probe.family == "consistency":
        completion = client.complete_pair(probe, sys_text, user_text, probe.user_b)
        text, text_b = completion.text, completion.text_b
    else:
        completion = client.complete(sys_text, user_text, probe=probe)
        text, text_b = completion.text, ""

    if probe is None:
        probe = Probe(
            id="freeform",
            family="sycophancy",
            split="adhoc",
            system=sys_text,
            user=user_text,
            clean_contains=(),
            spoiled_contains=(),
        )

    signals = measure_all(probe, text, text_b)
    gate = get_gate().predict(signals)
    latency_ms = int(round((time.perf_counter() - t0) * 1000.0))
    return {
        "probeId": probe.id,
        "family": probe.family,
        "split": probe.split,
        "backend": completion.backend,
        "live": completion.live,
        "contaminated": contaminate,
        "verdict": gate["verdict"],
        "score": gate["pCorrupt"],
        "signals": {
            name: {"value": round(pair["value"], 4), "score": round(pair["score"], 4)}
            for name, pair in signals.items()
        },
        "reasons": reasons(signals, gate),
        "completion": text,
        "completionB": text_b,
        "system": sys_text,
        "user": user_text,
        "baseline": gate["baseline"],
        "model": gate,
        "latencyMs": latency_ms,
    }

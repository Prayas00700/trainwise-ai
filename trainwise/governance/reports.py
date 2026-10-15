"""Markdown governance report (for reviewers, auditors and the model owners)."""
from __future__ import annotations

import time
from pathlib import Path

SIGNAL = {"green": "Clear (green)", "double_yellow": "Attention (double yellow)", "yellow": "Caution (yellow)",
          "red": "Danger (red)"}


def governance_report(p, cycle: dict | None = None) -> str:
    cycle = cycle or (p.supervisor.history[-1] if p.supervisor.history else None)
    lines = ["# Watchtower governance report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}; governance policy version {p.policies['version']}. "
             "All data in this prototype is synthetic.", "",
             "## Service health", "",
             "| Service | Signal | Version / mode | Requests | p95 ms | Live metric | Input drift |",
             "|---|---|---|---|---|---|---|"]
    for r in p.supervisor.health_board():
        lines.append(f"| {r['service']} | {SIGNAL[r['aspect']]} | {r['version']} | {r['requests']} | "
                     f"{r['p95_ms'] if r['p95_ms'] is not None else '-'} | {r['metric']} | {r['drift']} |")
    if cycle:
        lines += ["", f"## Latest governance cycle ({cycle['cycle_id']}, {cycle['ran_at']})", "", cycle["summary"], ""]
        for inc in cycle["incidents"]:
            rca = inc["rca"]
            lines += [f"### {inc['priority']}: {inc['target']} ({inc['status']})", "",
                      f"Root cause ({rca['confidence']:.0%} confidence): {rca['root_cause']}", "", "Findings:"]
            lines += [f"- {f['agent']}: {f['summary']}" for f in inc["findings"]]
            lines += ["", "Actions:"]
            for a in inc["actions"]:
                lines.append(f"- {a['action']}: {a['status']}" + (f" (verification: {a['verification']})" if a.get("verification") else ""))
            if rca["blast_radius"]:
                lines += ["", f"Downstream services exposed: {', '.join(rca['blast_radius'])}"]
            lines.append("")
    pending = [a for a in p.approvals if a["status"] == "pending"]
    lines += ["## Pending human approvals", ""]
    lines += [f"- {a['approval_id']}: {a['action']} {a['target']} {a.get('version') or ''} ({a['why']})" for a in pending] or ["None."]
    ver = p.audit.verify()
    lines += ["", "## Audit trail", "",
              f"{ver['entries']} entries; hash chain {'intact' if ver['valid'] else 'BROKEN at entry ' + str(ver['first_broken_seq'])}."]
    return "\n".join(lines) + "\n"


def save_report(p, text: str) -> Path:
    out = Path(p.artifact_dir) / "reports" / f"governance_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out

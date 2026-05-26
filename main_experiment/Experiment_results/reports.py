from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


FINAL_FIGURE_STEMS = [
    "fig3_main_comparison",
    "fig4_condition_switching_response",
    "fig5_implant_stress_response",
    "fig6_ablation_hfss",
    "fig7_phase_diagram_implant_shortage",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(item) for item in value)}]")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pass_fail_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def remove_final_figures(final_dir: Path) -> None:
    for stem in FINAL_FIGURE_STEMS:
        for suffix in [".pdf", ".png", ".csv"]:
            target = final_dir / f"{stem}{suffix}"
            if target.exists():
                target.unlink()


def write_failed_report(final_dir: Path, *, stage: str, failed_codes: list[str], reason: str) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    remove_final_figures(final_dir)
    lines = [
        "# failed_report",
        "",
        f"- stage: {stage}",
        f"- reason: {reason}",
        f"- failed gates: {', '.join(failed_codes) if failed_codes else 'none recorded'}",
        "",
        "## diagnosis",
        "- 没有输出 final figures，因为至少一个 hard/figure-level gate 未通过。",
        "- 这通常意味着工作点、stress severity、负载强度、预算范围或服务质量指标区分度仍需调整。",
        "- 不应把未通过 gate 的 quick figure 包装成论文终稿图。",
    ]
    (final_dir / "failed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


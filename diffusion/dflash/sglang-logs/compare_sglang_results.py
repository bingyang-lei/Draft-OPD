from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DATASET_RE = re.compile(r"^### Dataset: `(.+)`$")
MEASURED_PROMPTS_RE = re.compile(r"^- measured_prompts: `(.+)`$")
OUTPUT_TOKS_HEADER_RE = re.compile(r"^#### (?!Baseline\b|Speedup\b).+ output tok/s$")
ACCEPTANCE_HEADER_RE = re.compile(r"^#### .+ acceptance length$")
DEFAULT_RESULT_DIR = Path("sglang-logs/qwen3-8b-2")
DEFAULT_RESULTS_GLOB = "sglang_results_*.md"
DRAFT_INDEX_RE = re.compile(r"_(\d+)\.md$")


@dataclass
class DatasetMetrics:
    measured_prompts: str | None = None
    output_toks_per_s: dict[str, float] = field(default_factory=dict)
    acceptance_length: dict[str, float] = field(default_factory=dict)


@dataclass
class Report:
    path: Path
    label: str
    datasets: dict[str, DatasetMetrics]


def parse_num(text: str) -> float:
    return float(text.strip().replace(",", ""))


def parse_md_table_row(line: str) -> list[str]:
    line = line.strip()
    if not (line.startswith("|") and line.endswith("|")):
        return []
    return [cell.strip() for cell in line.strip("|").split("|")]


def infer_label(path: Path) -> str:
    match = re.search(r"(_\d+)\.md$", path.name)
    if match:
        return match.group(1) + ".md"
    return path.name


def parse_report(path: Path, label: str | None = None) -> Report:
    datasets: dict[str, DatasetMetrics] = {}
    current_dataset: str | None = None
    current_metric: str | None = None
    current_concs: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        dataset_match = DATASET_RE.match(line)
        if dataset_match:
            dataset_name = dataset_match.group(1)
            current_dataset = dataset_name
            datasets.setdefault(dataset_name, DatasetMetrics())
            current_metric = None
            current_concs = []
            continue

        if current_dataset is None:
            continue

        measured_match = MEASURED_PROMPTS_RE.match(line)
        if measured_match:
            datasets[current_dataset].measured_prompts = measured_match.group(1)
            continue

        if OUTPUT_TOKS_HEADER_RE.match(line):
            current_metric = "output_toks_per_s"
            current_concs = []
            continue
        if ACCEPTANCE_HEADER_RE.match(line):
            current_metric = "acceptance_length"
            current_concs = []
            continue

        if current_metric is None:
            continue

        cells = parse_md_table_row(line)
        if not cells:
            continue
        row_name = cells[0].lower()
        if row_name == "conc":
            current_concs = cells[1:]
        elif row_name == "value" and current_concs:
            values = cells[1:]
            for conc, value in zip(current_concs, values):
                getattr(datasets[current_dataset], current_metric)[conc] = parse_num(value)

    return Report(path=path, label=label or infer_label(path), datasets=datasets)


def ordered_datasets(reports: Iterable[Report]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for report in reports:
        for dataset in report.datasets:
            if dataset not in seen:
                seen.add(dataset)
                order.append(dataset)
    return order


def format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def metric_value(metrics: DatasetMetrics | None, metric: str, conc: str) -> float | None:
    if metrics is None:
        return None
    values: dict[str, float] = getattr(metrics, metric)
    return values.get(conc)


def max_min_growth_pct(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    min_value = min(values)
    if min_value <= 0:
        return None
    return (max(values) / min_value - 1) * 100


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def render_metric_table(
    reports: list[Report],
    datasets: list[str],
    *,
    metric: str,
    conc: str,
    digits: int,
) -> str:
    rows: list[list[str]] = []
    for dataset in datasets:
        measured = next(
            (
                report.datasets[dataset].measured_prompts
                for report in reports
                if dataset in report.datasets and report.datasets[dataset].measured_prompts
            ),
            "-",
        )
        row = [dataset, measured]
        for report in reports:
            row.append(format_float(metric_value(report.datasets.get(dataset), metric, conc), digits))
        rows.append(row)

    return render_table(["bench", "prompts", *[report.label for report in reports]], rows)


def render_combined_table(reports: list[Report], datasets: list[str], *, conc: str) -> str:
    rows: list[list[str]] = []
    for dataset in datasets:
        measured = next(
            (
                report.datasets[dataset].measured_prompts
                for report in reports
                if dataset in report.datasets and report.datasets[dataset].measured_prompts
            ),
            "-",
        )
        row = [dataset, measured]
        for report in reports:
            metrics = report.datasets.get(dataset)
            toks = metric_value(metrics, "output_toks_per_s", conc)
            accept = metric_value(metrics, "acceptance_length", conc)
            row.append(f"{format_float(toks, 2)} tok/s<br>{format_float(accept, 3)} accept")
        rows.append(row)
    return render_table(["bench", "prompts", *[report.label for report in reports]], rows)


def render_summary_table(reports: list[Report], datasets: list[str], *, conc: str) -> str:
    rows: list[list[str]] = []
    for dataset in datasets:
        toks = [
            (report.label, metric_value(report.datasets.get(dataset), "output_toks_per_s", conc))
            for report in reports
        ]
        accepts = [
            (report.label, metric_value(report.datasets.get(dataset), "acceptance_length", conc))
            for report in reports
        ]
        valid_toks = [(label, value) for label, value in toks if value is not None]
        valid_accepts = [(label, value) for label, value in accepts if value is not None]

        if valid_toks:
            best_toks_label, best_toks = max(valid_toks, key=lambda item: item[1])
            toks_growth_pct = max_min_growth_pct(value for _, value in valid_toks)
        else:
            best_toks_label, best_toks, toks_growth_pct = "-", None, None

        if valid_accepts:
            best_accept_label, best_accept = max(valid_accepts, key=lambda item: item[1])
            accept_growth_pct = max_min_growth_pct(value for _, value in valid_accepts)
        else:
            best_accept_label, best_accept, accept_growth_pct = "-", None, None

        rows.append(
            [
                dataset,
                f"{best_toks_label} ({format_float(best_toks, 2)})",
                f"{best_accept_label} ({format_float(best_accept, 3)})",
                "-" if toks_growth_pct is None else f"{toks_growth_pct:.2f}%",
                "-" if accept_growth_pct is None else f"{accept_growth_pct:.2f}%",
            ]
        )

    return render_table(
        ["bench", "best tok/s", "best acceptance", "tok/s max/min gain", "acc max/min gain"],
        rows,
    )


def render_markdown(reports: list[Report], *, conc: str) -> str:
    datasets = ordered_datasets(reports)
    files = "\n".join(f"- `{report.label}`: `{report.path}`" for report in reports)
    return "\n\n".join(
        [
            "# DFLASH Result Comparison",
            "## Files\n" + files,
            "## Combined\n" + render_combined_table(reports, datasets, conc=conc),
            "## Output Tok/S\n"
            + render_metric_table(
                reports,
                datasets,
                metric="output_toks_per_s",
                conc=conc,
                digits=2,
            ),
            "## Acceptance Length\n"
            + render_metric_table(
                reports,
                datasets,
                metric="acceptance_length",
                conc=conc,
                digits=3,
            ),
            "## Summary\n" + render_summary_table(reports, datasets, conc=conc),
        ]
    )


def default_report_sort_key(path: Path) -> tuple[int, str]:
    m = DRAFT_INDEX_RE.search(path.name)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def default_files() -> list[Path]:
    """All matching benchmark `.md` under DEFAULT_RESULT_DIR; count can be arbitrary."""
    here = DEFAULT_RESULT_DIR
    skip = default_output_md().name
    paths = [
        p
        for p in here.glob(DEFAULT_RESULTS_GLOB)
        if p.is_file() and p.name != skip
    ]
    return sorted(paths, key=default_report_sort_key)


def default_output_md() -> Path:
    return DEFAULT_RESULT_DIR / "compare_sglang_results.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DFLASH sglang markdown benchmark reports.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=(
            "Markdown result files. If omitted, uses all "
            f"`{DEFAULT_RESULTS_GLOB}` under `{DEFAULT_RESULT_DIR}` (any count)."
        ),
    )
    parser.add_argument(
        "--conc",
        default="1",
        help="Concurrency column to compare from each report table. Default: 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(args.files) if args.files else default_files()
    if not paths:
        raise SystemExit(
            f"No markdown inputs: pass files on the CLI or add `{DEFAULT_RESULTS_GLOB}` under "
            f"`{DEFAULT_RESULT_DIR}`."
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing result file(s): " + ", ".join(str(path) for path in missing))

    reports = [parse_report(path) for path in paths]
    markdown = render_markdown(reports, conc=str(args.conc))
    print(markdown)

    output_md = default_output_md()
    output_md.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

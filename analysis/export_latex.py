"""Auto-generate publication-grade booktabs LaTeX tables from benchmark CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.stats_tests import bootstrap_ci


def format_benchmark_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    method_col: str = "method",
    higher_is_better: dict[str, bool] | None = None,
    caption: str = "Cross-domain quantitative comparison.",
    label: str = "tab:main_results",
) -> str:
    """Create an Elsevier/IEEE booktabs LaTeX table with mean ± 95% CI, bold best & underlined second."""
    if higher_is_better is None:
        higher_is_better = {m: (m != "hd95") for m in metric_cols}

    methods = list(df[method_col].unique())
    summary: dict[str, dict[str, tuple[float, float, float]]] = {}
    for meth in methods:
        sub = df[df[method_col] == meth]
        summary[meth] = {m: bootstrap_ci(sub[m].to_numpy()) for m in metric_cols}

    # Determine rank 1 and rank 2 per metric
    best_second: dict[str, tuple[str, str]] = {}
    for m in metric_cols:
        ordered = sorted(
            methods,
            key=lambda x: summary[x][m][0],
            reverse=higher_is_better.get(m, True),
        )
        best_second[m] = (ordered[0], ordered[1] if len(ordered) > 1 else ordered[0])

    col_spec = "l" + "c" * len(metric_cols)
    headers = " & ".join([r"\textbf{Method}"] + [rf"\textbf{{{m.upper().replace('_', ' ')}}}" for m in metric_cols])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]

    for meth in methods:
        cells = [meth.replace("_", r"\_")]
        for m in metric_cols:
            mean_v, lo, hi = summary[meth][m]
            half = 0.5 * (hi - lo)
            txt = f"{mean_v:.3f} $\\pm$ {half:.3f}" if m != "hd95" else f"{mean_v:.2f} $\\pm$ {half:.2f}"
            b1, b2 = best_second[m]
            if meth == b1:
                txt = rf"\textbf{{{txt}}}"
            elif meth == b2 and b2 != b1:
                txt = rf"\underline{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LaTeX tables from benchmark CSV logs.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=["dsc", "iou", "hd95", "boundary_f1", "specificity"])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    tex = format_benchmark_table(df, metric_cols=args.metrics)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(tex)
    print(f"Exported LaTeX table to {args.out_tex}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


METRIC_KEYS = ["mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"]
METRIC_PATTERNS = {
    key: re.compile(rf"{key}:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    for key in METRIC_KEYS
}


def parse_metrics(log_file: Path):
    metrics = {key: "" for key in METRIC_KEYS}
    if not log_file.is_file():
        return metrics

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for key, pattern in METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    metrics[key] = match.group(1)
    return metrics


def load_manifest(manifest_file: Path):
    rows = []
    with manifest_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def write_csv(out_file: Path, rows):
    fieldnames = [
        "phase",
        "experiment",
        "status",
        "port",
        "config",
        "log_file",
        *METRIC_KEYS,
    ]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    header = ["phase", "experiment", "status", "mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"]
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    row["phase"],
                    row["experiment"],
                    row["status"],
                    row["mAP"],
                    row["NDS"],
                    row["mATE"],
                    row["mASE"],
                    row["mAOE"],
                    row["mAVE"],
                    row["mAAE"],
                ]
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Extract key metrics from ablation/scan logs.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.tsv from run_rwhi_ggf_ablation_scan.sh")
    parser.add_argument("--output", default="", help="Output csv file path (default: <manifest_dir>/metrics.csv)")
    parser.add_argument("--print-table", action="store_true", help="Print summary table to stdout")
    args = parser.parse_args()

    manifest_file = Path(args.manifest).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_file}")

    rows = load_manifest(manifest_file)
    merged_rows = []
    for row in rows:
        log_file = Path(row["log_file"])
        if log_file.is_absolute():
            resolved_log_file = log_file
        else:
            cwd_candidate = (Path.cwd() / log_file).resolve()
            manifest_candidate = (manifest_file.parent / log_file).resolve()
            if cwd_candidate.is_file():
                resolved_log_file = cwd_candidate
            elif manifest_candidate.is_file():
                resolved_log_file = manifest_candidate
            else:
                # Fall back to cwd-based resolution to keep output deterministic.
                resolved_log_file = cwd_candidate
        metrics = parse_metrics(resolved_log_file)
        merged_rows.append({**row, **metrics, "log_file": str(resolved_log_file)})

    if args.output:
        output_file = Path(args.output).resolve()
    else:
        output_file = manifest_file.parent / "metrics.csv"
    write_csv(output_file, merged_rows)

    print(f"Saved metrics csv: {output_file}")
    if args.print_table:
        print_table(merged_rows)


if __name__ == "__main__":
    main()

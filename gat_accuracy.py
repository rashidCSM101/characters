"""
GAT Graph Accuracy Calculator
==============================
Reads graph_data.json produced by:
    python gat_nastaleeq.py demo --image <img>

Computes node-level, edge-level (per type H/V/N), and overall accuracy
metrics against ground-truth counts you provide.

Usage
-----
# Provide ground-truth on command line:
    python gat_accuracy.py --json output_grouped/image_gat_graphs/graph_data.json
                           --gt_nodes 12
                           --gt_H 24 --gt_V 3 --gt_N 2

# Provide ground-truth as a JSON file:
    python gat_accuracy.py --json output_grouped/image_gat_graphs/graph_data.json
                           --gt_file ground_truth/image_gt.json

# Process ALL graph_data.json files under output_grouped at once:
    python gat_accuracy.py --all --gt_file ground_truth/all_gt.json

Ground-truth JSON format (single image):
    {
      "image": "image.jpg",
      "total_gt_nodes": 12,
      "total_gt_H": 24,
      "total_gt_V": 3,
      "total_gt_N": 2,
      "lines": [
        {"line": 1, "gt_nodes": 7, "gt_H": 16, "gt_V": 2, "gt_N": 0},
        {"line": 2, "gt_nodes": 5, "gt_H": 8,  "gt_V": 1, "gt_N": 2}
      ]
    }

Ground-truth JSON format (multiple images, for --all mode):
    [
      {"image": "image",   "total_gt_nodes": 12, "total_gt_H": 24, "total_gt_V": 3, "total_gt_N": 2},
      {"image": "img-1",   "total_gt_nodes": 13, "total_gt_H": 37, "total_gt_V": 14, "total_gt_N": 4}
    ]
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _prf(tp, fp, fn):
    """Return (precision, recall, f1) as percentages."""
    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def compute_count_metrics(detected, actual):
    """
    Count-based TP/FP/FN given detected and actual integer counts.
    TP = min(detected, actual)
    FP = max(0, detected - actual)   [over-detection]
    FN = max(0, actual - detected)   [missed]
    """
    tp = min(detected, actual)
    fp = max(0, detected - actual)
    fn = max(0, actual  - detected)
    accuracy = 100.0 * tp / actual if actual > 0 else (100.0 if detected == 0 else 0.0)
    prec, rec, f1 = _prf(tp, fp, fn)
    return dict(actual=actual, detected=detected,
                tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1, accuracy=accuracy)


def compute_graph_metrics(graph_json, gt_nodes, gt_H, gt_V, gt_N):
    """
    Compute all accuracy metrics for one graph_data.json content.

    Args:
        graph_json : parsed list from graph_data.json
        gt_nodes, gt_H, gt_V, gt_N : ground-truth integers (totals across lines)

    Returns:
        dict with per-line and aggregate metrics
    """
    # ── Aggregate detected values across all lines ────────────────────────
    det_nodes = sum(line['num_nodes'] for line in graph_json)
    det_H = sum(e['type'] == 'H' for line in graph_json for e in line['edges'])
    det_V = sum(e['type'] == 'V' for line in graph_json for e in line['edges'])
    det_N = sum(e['type'] == 'N' for line in graph_json for e in line['edges'])
    det_edges_total = det_H + det_V + det_N
    gt_edges_total  = gt_H + gt_V + gt_N

    # ── Per-category metrics ──────────────────────────────────────────────
    node_m  = compute_count_metrics(det_nodes, gt_nodes)
    edge_H  = compute_count_metrics(det_H, gt_H)
    edge_V  = compute_count_metrics(det_V, gt_V)
    edge_N  = compute_count_metrics(det_N, gt_N)
    edge_all = compute_count_metrics(det_edges_total, gt_edges_total)

    # ── Overall accuracy (nodes + edges combined) ─────────────────────────
    tp_total  = node_m['tp']  + edge_all['tp']
    fp_total  = node_m['fp']  + edge_all['fp']
    fn_total  = node_m['fn']  + edge_all['fn']
    gt_total  = gt_nodes + gt_edges_total
    det_total = det_nodes + det_edges_total

    overall_accuracy  = 100.0 * tp_total / gt_total if gt_total > 0 else 0.0
    overall_prec, overall_rec, overall_f1 = _prf(tp_total, fp_total, fn_total)

    # ── Edge type distribution accuracy ──────────────────────────────────
    # How close is the detected H:V:N ratio to the ground-truth ratio?
    def ratio(h, v, n):
        t = h + v + n
        return (h/t, v/t, n/t) if t > 0 else (0.0, 0.0, 0.0)

    gt_ratio  = ratio(gt_H,  gt_V,  gt_N)
    det_ratio = ratio(det_H, det_V, det_N)
    # Distribution error = mean absolute deviation of ratios (0-100 scale)
    dist_error = 100.0 * sum(abs(a-b) for a, b in zip(gt_ratio, det_ratio)) / 3.0
    dist_accuracy = max(0.0, 100.0 - dist_error)

    # ── Graph density ─────────────────────────────────────────────────────
    # density = edges / (N*(N-1)) for directed graph
    def density(n_nodes, n_edges):
        max_e = n_nodes * (n_nodes - 1)
        return n_edges / max_e if max_e > 0 else 0.0

    det_density = density(det_nodes, det_edges_total)
    gt_density  = density(gt_nodes,  gt_edges_total)

    # ── Per-line summary ──────────────────────────────────────────────────
    per_line = []
    for line in graph_json:
        lH = sum(1 for e in line['edges'] if e['type'] == 'H')
        lV = sum(1 for e in line['edges'] if e['type'] == 'V')
        lN = sum(1 for e in line['edges'] if e['type'] == 'N')
        per_line.append(dict(
            line        = line['line'],
            nodes       = line['num_nodes'],
            edges_H     = lH,
            edges_V     = lV,
            edges_N     = lN,
            edges_total = lH + lV + lN,
        ))

    return dict(
        # raw counts
        detected_nodes       = det_nodes,
        detected_edges_H     = det_H,
        detected_edges_V     = det_V,
        detected_edges_N     = det_N,
        detected_edges_total = det_edges_total,
        gt_nodes             = gt_nodes,
        gt_edges_H           = gt_H,
        gt_edges_V           = gt_V,
        gt_edges_N           = gt_N,
        gt_edges_total       = gt_edges_total,
        # per-category metrics
        node_metrics         = node_m,
        edge_H_metrics       = edge_H,
        edge_V_metrics       = edge_V,
        edge_N_metrics       = edge_N,
        edge_total_metrics   = edge_all,
        # overall
        tp_total             = tp_total,
        fp_total             = fp_total,
        fn_total             = fn_total,
        overall_accuracy     = overall_accuracy,
        overall_precision    = overall_prec,
        overall_recall       = overall_rec,
        overall_f1           = overall_f1,
        # distribution
        gt_edge_ratio        = dict(H=gt_ratio[0],  V=gt_ratio[1],  N=gt_ratio[2]),
        det_edge_ratio       = dict(H=det_ratio[0], V=det_ratio[1], N=det_ratio[2]),
        distribution_accuracy= dist_accuracy,
        dist_error_pct       = dist_error,
        # density
        detected_density     = det_density,
        gt_density           = gt_density,
        # per-line
        per_line             = per_line,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────────────────────────────────────

def bar(value, max_val=100.0, width=30, fill='#', empty='-'):
    """Simple ASCII progress bar for a percentage value."""
    filled = int(round(width * min(value, max_val) / max_val))
    return '[' + fill * filled + empty * (width - filled) + ']'


def print_report(image_name, m):
    """Print a formatted accuracy report to stdout."""
    sep  = '=' * 70
    sep2 = '-' * 70

    print()
    print(sep)
    print(f"  GAT ACCURACY REPORT: {image_name}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    # ── Detected vs Ground Truth counts ──────────────────────────────────
    print()
    print("  DETECTED vs GROUND TRUTH")
    print(sep2)
    print(f"  {'Category':<30} {'Detected':>10} {'GT':>10} {'Match'}")
    print(sep2)

    cats = [
        ("Nodes (ligature components)", m['detected_nodes'],       m['gt_nodes']),
        ("Edges - Horizontal (H)",      m['detected_edges_H'],     m['gt_edges_H']),
        ("Edges - Vertical   (V)",      m['detected_edges_V'],     m['gt_edges_V']),
        ("Edges - Nukta      (N)",      m['detected_edges_N'],     m['gt_edges_N']),
        ("Edges - TOTAL",               m['detected_edges_total'], m['gt_edges_total']),
    ]
    for label, det, gt in cats:
        match = 'EXACT' if det == gt else ('+%-d' % (det-gt) if det > gt else '%-d' % (det-gt))
        print(f"  {label:<30} {det:>10} {gt:>10}   {match}")

    # ── Per-category metrics table ─────────────────────────────────────────
    print()
    print("  METRICS PER CATEGORY")
    print(sep2)
    print(f"  {'Category':<22} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'Acc%':>7} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    print(sep2)

    metric_rows = [
        ("Nodes",          m['node_metrics']),
        ("Edges H",        m['edge_H_metrics']),
        ("Edges V",        m['edge_V_metrics']),
        ("Edges N",        m['edge_N_metrics']),
        ("Edges (total)",  m['edge_total_metrics']),
    ]
    for label, mx in metric_rows:
        print(f"  {label:<22} {mx['precision']:>7.2f} {mx['recall']:>7.2f} "
              f"{mx['f1']:>7.2f} {mx['accuracy']:>7.2f} "
              f"{mx['tp']:>5} {mx['fp']:>5} {mx['fn']:>5}")

    # ── Edge type distribution ─────────────────────────────────────────────
    print()
    print("  EDGE TYPE DISTRIBUTION")
    print(sep2)
    gr = m['gt_edge_ratio']
    dr = m['det_edge_ratio']
    for t in ('H', 'V', 'N'):
        print(f"  Type {t}: GT={gr[t]*100:5.1f}%  Detected={dr[t]*100:5.1f}%")
    print(f"  Distribution accuracy: {m['distribution_accuracy']:.2f}%  "
          f"(error={m['dist_error_pct']:.2f}%)")

    # ── Graph density ──────────────────────────────────────────────────────
    print()
    print("  GRAPH DENSITY")
    print(sep2)
    print(f"  Detected density: {m['detected_density']:.4f}")
    print(f"  GT density:       {m['gt_density']:.4f}")

    # ── Per-line breakdown ─────────────────────────────────────────────────
    if m['per_line']:
        print()
        print("  PER-LINE BREAKDOWN (detected)")
        print(sep2)
        print(f"  {'Line':>5} {'Nodes':>6} {'H':>6} {'V':>6} {'N':>6} {'Total':>7}")
        print(sep2)
        for pl in m['per_line']:
            print(f"  {pl['line']:>5} {pl['nodes']:>6} {pl['edges_H']:>6} "
                  f"{pl['edges_V']:>6} {pl['edges_N']:>6} {pl['edges_total']:>7}")

    # ── Final summary box ──────────────────────────────────────────────────
    print()
    print(sep)
    print("  FINAL RESULT")
    print(sep)

    metrics_summary = [
        ("Overall Accuracy",         m['overall_accuracy']),
        ("Overall Precision",        m['overall_precision']),
        ("Overall Recall",           m['overall_recall']),
        ("Overall F1-Score",         m['overall_f1']),
        ("Node Detection Accuracy",  m['node_metrics']['accuracy']),
        ("Edge Detection Accuracy",  m['edge_total_metrics']['accuracy']),
        ("Edge Distribution Acc.",   m['distribution_accuracy']),
    ]
    for label, val in metrics_summary:
        print(f"  {label:<30} {val:>7.2f}%  {bar(val)}")

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"graph_data.json not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_report_json(metrics, out_path):
    """Save the full metrics dict to out_path as JSON."""
    # Make all values JSON-serialisable (round floats for readability)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_clean(metrics), f, indent=2, ensure_ascii=False)
    print(f"  [ok] Accuracy report saved: {out_path}")


def _gt_from_args(args):
    """Build gt scalars from CLI args (--gt_nodes, --gt_H, --gt_V, --gt_N)."""
    missing = [a for a in ['gt_nodes', 'gt_H', 'gt_V', 'gt_N']
               if getattr(args, a, None) is None]
    if missing:
        raise ValueError(
            f"Missing ground-truth args: {', '.join('--'+m for m in missing)}\n"
            "Either supply --gt_nodes/--gt_H/--gt_V/--gt_N  OR  --gt_file"
        )
    return args.gt_nodes, args.gt_H, args.gt_V, args.gt_N


# ─────────────────────────────────────────────────────────────────────────────
# Single-image mode
# ─────────────────────────────────────────────────────────────────────────────

def run_single(json_path, gt_nodes, gt_H, gt_V, gt_N, image_name=None):
    graph_json = load_graph_json(json_path)

    if image_name is None:
        # Infer from folder name: output_grouped/<name>_gat_graphs/graph_data.json
        image_name = os.path.basename(os.path.dirname(json_path))

    print(f"\nLoaded: {json_path}")
    print(f"  Lines in file : {len(graph_json)}")
    tot_nodes = sum(l['num_nodes'] for l in graph_json)
    tot_edges = sum(len(l['edges']) for l in graph_json)
    print(f"  Total nodes   : {tot_nodes}")
    print(f"  Total edges   : {tot_edges}")

    metrics = compute_graph_metrics(graph_json, gt_nodes, gt_H, gt_V, gt_N)
    print_report(image_name, metrics)

    # Save next to the input JSON
    out_dir  = os.path.dirname(json_path)
    out_path = os.path.join(out_dir, 'accuracy_report.json')
    save_report_json(dict(image=image_name,
                          gt=dict(nodes=gt_nodes, H=gt_H, V=gt_V, N=gt_N),
                          metrics=metrics,
                          computed_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                     out_path)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Batch (--all) mode
# ─────────────────────────────────────────────────────────────────────────────

def run_all(output_grouped_dir, gt_list):
    """
    Find all graph_data.json files under output_grouped_dir and evaluate each.
    gt_list is a list of dicts keyed by 'image'.
    """
    gt_map = {entry['image']: entry for entry in gt_list}

    json_files = []
    for root, dirs, files in os.walk(output_grouped_dir):
        for fn in files:
            if fn == 'graph_data.json':
                json_files.append(os.path.join(root, fn))

    if not json_files:
        print(f"No graph_data.json files found under: {output_grouped_dir}")
        return

    all_metrics = []
    for jf in sorted(json_files):
        folder_name = os.path.basename(os.path.dirname(jf))
        # Strip _gat_graphs suffix to recover image base name
        img_name = folder_name.replace('_gat_graphs', '')

        if img_name not in gt_map:
            print(f"\n[skip] No GT entry for '{img_name}' in GT file. Skipping {jf}")
            continue

        gt_entry = gt_map[img_name]
        m = run_single(
            jf,
            gt_entry.get('total_gt_nodes', 0),
            gt_entry.get('total_gt_H', 0),
            gt_entry.get('total_gt_V', 0),
            gt_entry.get('total_gt_N', 0),
            image_name=img_name
        )
        all_metrics.append(dict(image=img_name, metrics=m))

    if not all_metrics:
        return

    # ── Aggregate across all images ────────────────────────────────────────
    keys = ['overall_accuracy', 'overall_precision', 'overall_recall',
            'overall_f1', 'distribution_accuracy']
    print()
    print('=' * 70)
    print('  AGGREGATE ACROSS ALL IMAGES')
    print('=' * 70)
    for key in keys:
        vals = [am['metrics'][key] for am in all_metrics]
        avg = sum(vals) / len(vals)
        mn  = min(vals)
        mx  = max(vals)
        print(f"  {key:<35} avg={avg:6.2f}%  min={mn:6.2f}%  max={mx:6.2f}%")
    print('=' * 70)

    # Save aggregate report
    agg_path = os.path.join(output_grouped_dir, 'aggregate_accuracy.json')
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(dict(images=all_metrics,
                       computed_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                  f, indent=2, ensure_ascii=False)
    print(f"\n  [ok] Aggregate report saved: {agg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute accuracy of GAT graph_data.json results',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Single image, GT on command line:
  python gat_accuracy.py --json output_grouped/image_gat_graphs/graph_data.json
                         --gt_nodes 12 --gt_H 22 --gt_V 2 --gt_N 0

  # Single image, GT from file:
  python gat_accuracy.py --json output_grouped/img-1_gat_graphs/graph_data.json
                         --gt_file ground_truth/img-1_gt.json

  # All images in output_grouped at once:
  python gat_accuracy.py --all --gt_file ground_truth/all_gt.json
"""
    )

    parser.add_argument('--json',     metavar='PATH',
                        help='Path to graph_data.json (single-image mode)')
    parser.add_argument('--gt_nodes', type=int, metavar='N',
                        help='Ground-truth node count')
    parser.add_argument('--gt_H',    type=int, metavar='N',
                        help='Ground-truth horizontal edge count')
    parser.add_argument('--gt_V',    type=int, metavar='N',
                        help='Ground-truth vertical edge count')
    parser.add_argument('--gt_N',    type=int, metavar='N',
                        help='Ground-truth nukta edge count')
    parser.add_argument('--gt_file', metavar='PATH',
                        help='JSON file with ground-truth (see format in docstring)')
    parser.add_argument('--all',     action='store_true',
                        help='Batch mode: process all graph_data.json under output_grouped/')
    parser.add_argument('--output_dir', default='output_grouped',
                        help='Root dir for --all mode (default: output_grouped)')

    args = parser.parse_args()

    # ── Resolve ground-truth source ────────────────────────────────────────
    if args.gt_file:
        if not os.path.exists(args.gt_file):
            print(f"ERROR: GT file not found: {args.gt_file}")
            sys.exit(1)
        with open(args.gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
    else:
        gt_data = None

    # ── Batch mode ─────────────────────────────────────────────────────────
    if args.all:
        if gt_data is None:
            print("ERROR: --all requires --gt_file")
            sys.exit(1)
        gt_list = gt_data if isinstance(gt_data, list) else [gt_data]
        run_all(args.output_dir, gt_list)
        return

    # ── Single mode ────────────────────────────────────────────────────────
    if not args.json:
        parser.print_help()
        print("\nERROR: --json is required in single-image mode")
        sys.exit(1)

    if gt_data is not None:
        # Single image GT file
        if isinstance(gt_data, list):
            gt_data = gt_data[0]
        gt_nodes = gt_data.get('total_gt_nodes', gt_data.get('gt_nodes', 0))
        gt_H     = gt_data.get('total_gt_H',     gt_data.get('gt_H', 0))
        gt_V     = gt_data.get('total_gt_V',     gt_data.get('gt_V', 0))
        gt_N     = gt_data.get('total_gt_N',     gt_data.get('gt_N', 0))
    else:
        try:
            gt_nodes, gt_H, gt_V, gt_N = _gt_from_args(args)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    run_single(args.json, gt_nodes, gt_H, gt_V, gt_N)


if __name__ == '__main__':
    main()

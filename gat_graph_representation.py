"""
GAT Graph Representation & Visualization
==========================================
Generates publication-quality visual representations of the spatial graph
constructed by gat_nastaleeq.py (Algorithm 1 from the paper).

Outputs:
    1. graph_overlay.png       - Typed edges drawn on the original image
    2. line_XX_graph.png       - Per-line zoomed graph overlays
    3. adjacency_matrices.png  - H/V/N adjacency matrix heatmaps
    4. node_link_diagram.png   - Abstract node-link diagram (spatial layout)
    5. edge_statistics.png     - Bar chart + pie chart of edge types
    6. full_dashboard.png      - Combined multi-panel figure with all views

Usage:
    python gat_graph_representation.py --image input_images/img-1.jpg

    # Custom output directory:
    python gat_graph_representation.py --image input_images/img-1.jpg --outdir my_graphs

    # Process all images in a folder:
    python gat_graph_representation.py --folder input_images
"""

import os
import sys
import argparse
import json
import math
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving only
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

# Import preprocessing + graph construction from gat_nastaleeq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gat_nastaleeq import (
    NastaleeqPreprocessor,
    NastaleeqGraphConstructor,
    EDGE_H, EDGE_V, EDGE_N,
    EDGE_LABELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Colour scheme  (consistent across all plots)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    EDGE_H: '#FF8C00',    # orange  — horizontal (reading order)
    EDGE_V: '#9D00FF',    # purple  — vertical   (stacking)
    EDGE_N: '#00CC44',    # green   — nukta-to-base
}
EDGE_LONG = {
    EDGE_H: 'Horizontal (H)',
    EDGE_V: 'Vertical (V)',
    EDGE_N: 'Nukta (N)',
}
NODE_COLOR      = '#2196F3'   # blue
NUKTA_COLOR     = '#FFEB3B'   # yellow
NODE_BORDER     = '#0D47A1'
NUKTA_THRESHOLD = 150          # px^2 area threshold (Table 1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Graph Overlay on Original Image
# ─────────────────────────────────────────────────────────────────────────────

def plot_graph_overlay(original_bgr, all_lines, all_graphs, out_path):
    """
    Draw bounding boxes and typed edges on the full original image.
    """
    img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(max(14, w / 80), max(6, h / 80)),
                           dpi=150)
    ax.imshow(img_rgb, aspect='equal')

    for line_data, graph in zip(all_lines, all_graphs):
        comps = line_data['components']
        ei    = graph['edge_index']
        et    = graph['edge_type']

        # Draw edges first (behind nodes)
        for k in range(ei.shape[1]):
            i, j  = int(ei[0, k]), int(ei[1, k])
            etype = int(et[k])
            c1 = (comps[i]['cx'], comps[i]['cy_global'])
            c2 = (comps[j]['cx'], comps[j]['cy_global'])
            ax.annotate('', xy=c2, xytext=c1,
                        arrowprops=dict(arrowstyle='->', color=COLORS[etype],
                                        lw=1.2, alpha=0.7))

        # Draw nodes (bounding boxes + centroid dots)
        for c in comps:
            is_nukta = c['area'] < NUKTA_THRESHOLD
            color = NUKTA_COLOR if is_nukta else NODE_COLOR
            rect = mpatches.FancyBboxPatch(
                (c['x'], c['y_global']), c['w'], c['h'],
                boxstyle='round,pad=1', linewidth=1.0,
                edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            ax.plot(c['cx'], c['cy_global'], 'o', color=color,
                    markersize=4, markeredgecolor=NODE_BORDER,
                    markeredgewidth=0.5)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=COLORS[EDGE_H], lw=2, label=EDGE_LONG[EDGE_H]),
        Line2D([0], [0], color=COLORS[EDGE_V], lw=2, label=EDGE_LONG[EDGE_V]),
        Line2D([0], [0], color=COLORS[EDGE_N], lw=2, label=EDGE_LONG[EDGE_N]),
        Line2D([0], [0], marker='s', color='w', markeredgecolor=NODE_COLOR,
               markerfacecolor='none', markersize=8, label='Ligature node'),
        Line2D([0], [0], marker='s', color='w', markeredgecolor=NUKTA_COLOR,
               markerfacecolor='none', markersize=8, label='Nukta node'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=7,
              framealpha=0.85, fancybox=True)
    ax.set_title('Spatial Graph Overlay (all lines)', fontsize=10, fontweight='bold')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-line Zoomed Graph
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_line_graph(line_binary, comps, graph, line_idx, out_path):
    """
    Zoomed graph overlay on a single line image.
    """
    line_rgb = cv2.cvtColor(line_binary, cv2.COLOR_GRAY2RGB)
    h, w = line_rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(max(10, w / 60), max(3, h / 60)),
                           dpi=150)
    ax.imshow(line_rgb, aspect='equal', cmap='gray')

    ei = graph['edge_index']
    et = graph['edge_type']

    # Edges
    for k in range(ei.shape[1]):
        i, j  = int(ei[0, k]), int(ei[1, k])
        etype = int(et[k])
        c1 = (comps[i]['cx'], comps[i]['cy'])
        c2 = (comps[j]['cx'], comps[j]['cy'])
        ax.annotate('', xy=c2, xytext=c1,
                    arrowprops=dict(arrowstyle='->', color=COLORS[etype],
                                    lw=1.5, alpha=0.8))

    # Nodes with index labels
    for idx, c in enumerate(comps):
        is_nukta = c['area'] < NUKTA_THRESHOLD
        color = NUKTA_COLOR if is_nukta else NODE_COLOR
        rect = mpatches.FancyBboxPatch(
            (c['x'], c['y']), c['w'], c['h'],
            boxstyle='round,pad=1', linewidth=1.2,
            edgecolor=color, facecolor='none', alpha=0.9
        )
        ax.add_patch(rect)
        ax.plot(c['cx'], c['cy'], 'o', color=color, markersize=6,
                markeredgecolor=NODE_BORDER, markeredgewidth=0.7)
        ax.text(c['cx'], c['cy'] - c['h'] * 0.6, str(idx),
                fontsize=6, ha='center', va='bottom', fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.6))

    # Edge type counts
    n_H = int(np.sum(et == EDGE_H))
    n_V = int(np.sum(et == EDGE_V))
    n_N = int(np.sum(et == EDGE_N))
    info = f"Line {line_idx}  |  {len(comps)} nodes  |  H={n_H}  V={n_V}  N={n_N}"
    ax.set_title(info, fontsize=9, fontweight='bold')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Adjacency Matrices (H / V / N heatmaps)
# ─────────────────────────────────────────────────────────────────────────────

def plot_adjacency_matrices(all_lines, all_graphs, out_path):
    """
    Three side-by-side adjacency matrix heatmaps, one per edge type.
    Combined across all lines (global node numbering).
    """
    # Build global adjacency matrices
    total_nodes = sum(len(ld['components']) for ld in all_lines)
    if total_nodes == 0:
        return

    adj_H = np.zeros((total_nodes, total_nodes), dtype=np.float32)
    adj_V = np.zeros((total_nodes, total_nodes), dtype=np.float32)
    adj_N = np.zeros((total_nodes, total_nodes), dtype=np.float32)

    offset = 0
    for ld, g in zip(all_lines, all_graphs):
        n  = len(ld['components'])
        ei = g['edge_index']
        et = g['edge_type']
        for k in range(ei.shape[1]):
            src = int(ei[0, k]) + offset
            dst = int(ei[1, k]) + offset
            etype = int(et[k])
            if etype == EDGE_H:
                adj_H[src, dst] = 1.0
            elif etype == EDGE_V:
                adj_V[src, dst] = 1.0
            elif etype == EDGE_N:
                adj_N[src, dst] = 1.0
        offset += n

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    matrices = [(adj_H, 'Horizontal (H)', '#FF8C00'),
                (adj_V, 'Vertical (V)',    '#9D00FF'),
                (adj_N, 'Nukta (N)',       '#00CC44')]

    for ax, (mat, title, color) in zip(axes, matrices):
        cmap = ListedColormap(['#FFFFFF', color])
        ax.imshow(mat, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Target node', fontsize=8)
        ax.set_ylabel('Source node', fontsize=8)
        ax.tick_params(labelsize=6)
        # Grid
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#CCCCCC')
        n_edges = int(mat.sum())
        ax.text(0.5, -0.12, f'{n_edges} edges', transform=ax.transAxes,
                fontsize=8, ha='center', color=color, fontweight='bold')

    fig.suptitle('Adjacency Matrices by Edge Type', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Abstract Node-Link Diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_node_link_diagram(all_lines, all_graphs, out_path):
    """
    Spatial node-link diagram: nodes placed by their (cx, cy_global) coords.
    Edges drawn as coloured arcs. Node size proportional to component area.
    """
    # Collect global node info
    nodes_cx  = []
    nodes_cy  = []
    nodes_area = []
    nodes_nukta = []
    node_labels = []

    edges_src = []
    edges_dst = []
    edges_type = []

    offset = 0
    for ld, g in zip(all_lines, all_graphs):
        comps = ld['components']
        for c in comps:
            nodes_cx.append(c['cx'])
            nodes_cy.append(c['cy_global'])
            nodes_area.append(c['area'])
            nodes_nukta.append(c['area'] < NUKTA_THRESHOLD)
            node_labels.append(str(offset + len(nodes_cx) - offset - 1 + offset))

        ei = g['edge_index']
        et = g['edge_type']
        for k in range(ei.shape[1]):
            edges_src.append(int(ei[0, k]) + offset)
            edges_dst.append(int(ei[1, k]) + offset)
            edges_type.append(int(et[k]))
        offset += len(comps)

    if not nodes_cx:
        return

    # Normalise positions for cleaner layout
    cx_arr = np.array(nodes_cx, dtype=np.float32)
    cy_arr = np.array(nodes_cy, dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=150)

    # Draw edges
    for s, d, et in zip(edges_src, edges_dst, edges_type):
        x1, y1 = cx_arr[s], cy_arr[s]
        x2, y2 = cx_arr[d], cy_arr[d]

        # Slight arc for overlapping edges
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        # Offset the midpoint perpendicular to the edge direction
        perp_scale = 0.05
        mid_x += -dy * perp_scale
        mid_y += dx * perp_scale

        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS[et],
                                    lw=1.0, alpha=0.6,
                                    connectionstyle='arc3,rad=0.1'))

    # Draw nodes
    for i in range(len(nodes_cx)):
        is_nk = nodes_nukta[i]
        color = NUKTA_COLOR if is_nk else NODE_COLOR
        size  = max(6, min(20, math.sqrt(nodes_area[i]) / 2))
        ax.plot(cx_arr[i], cy_arr[i], 'o', color=color, markersize=size,
                markeredgecolor=NODE_BORDER, markeredgewidth=0.8, alpha=0.85)
        ax.text(cx_arr[i], cy_arr[i], str(i), fontsize=5, ha='center',
                va='center', fontweight='bold', color='white')

    # RTL reading order arrow
    ax.annotate('RTL reading direction', xy=(cx_arr.min(), cy_arr.min() - 15),
                xytext=(cx_arr.max(), cy_arr.min() - 15),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                fontsize=7, color='gray', ha='right', va='bottom')

    # Legend
    legend_handles = [
        Line2D([0], [0], color=COLORS[EDGE_H], lw=2, label=EDGE_LONG[EDGE_H]),
        Line2D([0], [0], color=COLORS[EDGE_V], lw=2, label=EDGE_LONG[EDGE_V]),
        Line2D([0], [0], color=COLORS[EDGE_N], lw=2, label=EDGE_LONG[EDGE_N]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLOR,
               markersize=8, label='Ligature'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NUKTA_COLOR,
               markersize=8, label='Nukta'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=7,
              framealpha=0.9)

    ax.set_title('Abstract Node-Link Diagram (spatial layout)', fontsize=11,
                 fontweight='bold')
    ax.set_xlabel('x position (px)', fontsize=8)
    ax.set_ylabel('y position (px)', fontsize=8)
    ax.invert_yaxis()   # match image coordinate system
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Edge Statistics Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_edge_statistics(all_lines, all_graphs, out_path):
    """
    Two-panel figure:
      Left:  bar chart of edge counts per line per type
      Right: overall pie chart of edge type proportions
    """
    per_line_H = []
    per_line_V = []
    per_line_N = []
    line_labels = []

    for idx, (ld, g) in enumerate(zip(all_lines, all_graphs), 1):
        et = g['edge_type']
        per_line_H.append(int(np.sum(et == EDGE_H)))
        per_line_V.append(int(np.sum(et == EDGE_V)))
        per_line_N.append(int(np.sum(et == EDGE_N)))
        line_labels.append(f'L{idx}')

    total_H = sum(per_line_H)
    total_V = sum(per_line_V)
    total_N = sum(per_line_N)
    total_nodes = sum(len(ld['components']) for ld in all_lines)
    total_edges = total_H + total_V + total_N

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150,
                                    gridspec_kw={'width_ratios': [2, 1]})

    # ── Left: stacked bar chart ──────────────────────────────────────────
    x = np.arange(len(line_labels))
    bar_width = 0.5 if len(line_labels) < 8 else 0.7

    bars_H = ax1.bar(x, per_line_H, bar_width, label=EDGE_LONG[EDGE_H],
                     color=COLORS[EDGE_H], alpha=0.85)
    bars_V = ax1.bar(x, per_line_V, bar_width, bottom=per_line_H,
                     label=EDGE_LONG[EDGE_V], color=COLORS[EDGE_V], alpha=0.85)
    bottom_N = [h + v for h, v in zip(per_line_H, per_line_V)]
    bars_N = ax1.bar(x, per_line_N, bar_width, bottom=bottom_N,
                     label=EDGE_LONG[EDGE_N], color=COLORS[EDGE_N], alpha=0.85)

    # Value labels on bars
    for i in range(len(line_labels)):
        total_i = per_line_H[i] + per_line_V[i] + per_line_N[i]
        ax1.text(i, total_i + 0.5, str(total_i), ha='center', fontsize=7,
                 fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(line_labels, fontsize=8)
    ax1.set_ylabel('Number of edges', fontsize=9)
    ax1.set_title('Edge Counts per Line (stacked)', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.grid(axis='y', alpha=0.3)

    # Summary text below the bar chart
    summary = (f'Total: {total_nodes} nodes, {total_edges} edges  |  '
               f'H={total_H}  V={total_V}  N={total_N}  |  '
               f'Avg degree={2*total_edges/max(total_nodes,1):.1f}')
    ax1.text(0.5, -0.10, summary, transform=ax1.transAxes,
             fontsize=7, ha='center', color='#555555')

    # ── Right: pie chart ─────────────────────────────────────────────────
    sizes  = [total_H, total_V, total_N]
    labels = [f'H ({total_H})', f'V ({total_V})', f'N ({total_N})']
    colors_pie = [COLORS[EDGE_H], COLORS[EDGE_V], COLORS[EDGE_N]]

    # Filter out zeros
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors_pie) if s > 0]
    if non_zero:
        sz, lb, co = zip(*non_zero)
        wedges, texts, autotexts = ax2.pie(
            sz, labels=lb, colors=co, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 8}
        )
        for t in autotexts:
            t.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, 'No edges', ha='center', va='center', fontsize=10)

    ax2.set_title('Edge Type Distribution', fontsize=10, fontweight='bold')

    fig.suptitle('Graph Edge Statistics', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full Dashboard (multi-panel combined view)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_dashboard(original_bgr, all_lines, all_graphs, image_name, out_path):
    """
    Combined 2x3 panel figure with all key graph representations.
    """
    img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(20, 12), dpi=150)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    total_nodes = sum(len(ld['components']) for ld in all_lines)

    # ── Panel 1: Original image with graph overlay ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb, aspect='equal')
    for ld, g in zip(all_lines, all_graphs):
        comps = ld['components']
        ei, et = g['edge_index'], g['edge_type']
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            etype = int(et[k])
            ax1.annotate('', xy=(comps[j]['cx'], comps[j]['cy_global']),
                         xytext=(comps[i]['cx'], comps[i]['cy_global']),
                         arrowprops=dict(arrowstyle='->', color=COLORS[etype],
                                         lw=0.8, alpha=0.7))
        for c in comps:
            is_nk = c['area'] < NUKTA_THRESHOLD
            clr = NUKTA_COLOR if is_nk else NODE_COLOR
            ax1.plot(c['cx'], c['cy_global'], 'o', color=clr, markersize=3,
                     markeredgecolor=NODE_BORDER, markeredgewidth=0.3)
    ax1.set_title('(a) Graph Overlay', fontsize=9, fontweight='bold')
    ax1.axis('off')

    # ── Panel 2: Node-link diagram ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    all_cx, all_cy = [], []
    offset = 0
    for ld, g in zip(all_lines, all_graphs):
        comps = ld['components']
        for c in comps:
            all_cx.append(c['cx'])
            all_cy.append(c['cy_global'])
        ei, et = g['edge_index'], g['edge_type']
        for k in range(ei.shape[1]):
            s, d = int(ei[0, k]) + offset, int(ei[1, k]) + offset
            # will draw after collecting all nodes
        offset += len(comps)

    # Re-draw edges and nodes
    offset = 0
    for ld, g in zip(all_lines, all_graphs):
        comps = ld['components']
        ei, et = g['edge_index'], g['edge_type']
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            etype = int(et[k])
            ax2.annotate('', xy=(comps[j]['cx'], comps[j]['cy_global']),
                         xytext=(comps[i]['cx'], comps[i]['cy_global']),
                         arrowprops=dict(arrowstyle='->', color=COLORS[etype],
                                         lw=0.8, alpha=0.6,
                                         connectionstyle='arc3,rad=0.08'))
        for idx, c in enumerate(comps):
            is_nk = c['area'] < NUKTA_THRESHOLD
            clr = NUKTA_COLOR if is_nk else NODE_COLOR
            sz = max(4, min(12, math.sqrt(c['area']) / 3))
            ax2.plot(c['cx'], c['cy_global'], 'o', color=clr, markersize=sz,
                     markeredgecolor=NODE_BORDER, markeredgewidth=0.5, alpha=0.85)
            ax2.text(c['cx'], c['cy_global'], str(idx + offset), fontsize=4,
                     ha='center', va='center', color='white', fontweight='bold')
        offset += len(comps)

    ax2.invert_yaxis()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.15)
    ax2.set_title('(b) Node-Link Diagram', fontsize=9, fontweight='bold')

    # ── Panel 3: Adjacency matrices ──────────────────────────────────────
    ax3_sub = fig.add_subplot(gs[0, 2])
    # Combined adjacency matrix with edge-type colour coding
    if total_nodes > 0:
        combined = np.zeros((total_nodes, total_nodes, 3), dtype=np.float32)
        # White background
        combined[:] = 1.0
        offset = 0
        for ld, g in zip(all_lines, all_graphs):
            n = len(ld['components'])
            ei, et = g['edge_index'], g['edge_type']
            for k in range(ei.shape[1]):
                s = int(ei[0, k]) + offset
                d = int(ei[1, k]) + offset
                etype = int(et[k])
                # Parse hex color
                hx = COLORS[etype].lstrip('#')
                r, gg, b = int(hx[0:2], 16)/255, int(hx[2:4], 16)/255, int(hx[4:6], 16)/255
                combined[s, d] = [r, gg, b]
            offset += n
        ax3_sub.imshow(combined, interpolation='nearest')
    ax3_sub.set_title('(c) Combined Adjacency\n(H=orange V=purple N=green)',
                      fontsize=8, fontweight='bold')
    ax3_sub.set_xlabel('Target', fontsize=7)
    ax3_sub.set_ylabel('Source', fontsize=7)
    ax3_sub.tick_params(labelsize=5)

    # ── Panel 4: Edge type bar chart ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    per_line_H, per_line_V, per_line_N, line_labels = [], [], [], []
    for idx, (ld, g) in enumerate(zip(all_lines, all_graphs), 1):
        et = g['edge_type']
        per_line_H.append(int(np.sum(et == EDGE_H)))
        per_line_V.append(int(np.sum(et == EDGE_V)))
        per_line_N.append(int(np.sum(et == EDGE_N)))
        line_labels.append(f'L{idx}')

    x = np.arange(len(line_labels))
    bw = 0.5 if len(line_labels) < 8 else 0.7
    ax4.bar(x, per_line_H, bw, label='H', color=COLORS[EDGE_H], alpha=0.85)
    ax4.bar(x, per_line_V, bw, bottom=per_line_H, label='V',
            color=COLORS[EDGE_V], alpha=0.85)
    bn = [h + v for h, v in zip(per_line_H, per_line_V)]
    ax4.bar(x, per_line_N, bw, bottom=bn, label='N',
            color=COLORS[EDGE_N], alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(line_labels, fontsize=7)
    ax4.set_ylabel('Edges', fontsize=8)
    ax4.set_title('(d) Edges per Line', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=6)
    ax4.grid(axis='y', alpha=0.2)

    # ── Panel 5: Pie chart ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    tH, tV, tN = sum(per_line_H), sum(per_line_V), sum(per_line_N)
    sizes = [tH, tV, tN]
    labels_pie = [f'H ({tH})', f'V ({tV})', f'N ({tN})']
    colors_pie = [COLORS[EDGE_H], COLORS[EDGE_V], COLORS[EDGE_N]]
    nz = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 0]
    if nz:
        sz, lb, co = zip(*nz)
        ax5.pie(sz, labels=lb, colors=co, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 7})
    ax5.set_title('(e) Edge Type Ratio', fontsize=9, fontweight='bold')

    # ── Panel 6: Summary statistics table ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    total_edges = tH + tV + tN
    avg_deg = 2 * total_edges / max(total_nodes, 1)
    density = total_edges / (total_nodes * max(total_nodes - 1, 1))
    nukta_count = sum(1 for ld in all_lines for c in ld['components']
                      if c['area'] < NUKTA_THRESHOLD)

    stats_data = [
        ['Total lines',      str(len(all_lines))],
        ['Total nodes',      str(total_nodes)],
        ['  - Ligature',     str(total_nodes - nukta_count)],
        ['  - Nukta',        str(nukta_count)],
        ['Total edges',      str(total_edges)],
        ['  - H (horiz)',    str(tH)],
        ['  - V (vert)',     str(tV)],
        ['  - N (nukta)',    str(tN)],
        ['Avg degree',       f'{avg_deg:.2f}'],
        ['Graph density',    f'{density:.4f}'],
    ]

    table = ax6.table(cellText=stats_data,
                      colLabels=['Metric', 'Value'],
                      colWidths=[0.55, 0.35],
                      loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#F5F5F5' if row % 2 == 0 else 'white')
        cell.set_edgecolor('#CCCCCC')

    ax6.set_title('(f) Graph Summary', fontsize=9, fontweight='bold')

    fig.suptitle(f'GAT Graph Analysis: {image_name}', fontsize=13,
                 fontweight='bold', y=0.98)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [ok] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path, out_dir):
    """Run full graph representation pipeline for one image."""
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing: {image_path}")

    prep = NastaleeqPreprocessor()
    gc   = NastaleeqGraphConstructor()

    lines_data, binary, original = prep.process(image_path)
    print(f"  -> {len(lines_data)} text line(s) detected")

    all_graphs = []
    for ld in lines_data:
        g = gc.build_graph(ld['components'])
        all_graphs.append(g)

    total_nodes = sum(len(ld['components']) for ld in lines_data)
    total_edges = sum(g['edge_type'].shape[0] for g in all_graphs)
    print(f"  -> {total_nodes} nodes, {total_edges} edges total")

    # 1. Full overlay
    plot_graph_overlay(original, lines_data, all_graphs,
                       os.path.join(out_dir, 'graph_overlay.png'))

    # 2. Per-line graphs
    for idx, (ld, g) in enumerate(zip(lines_data, all_graphs), 1):
        plot_per_line_graph(ld['line_binary'], ld['components'], g, idx,
                           os.path.join(out_dir, f'line_{idx:02d}_graph.png'))

    # 3. Adjacency matrices
    plot_adjacency_matrices(lines_data, all_graphs,
                            os.path.join(out_dir, 'adjacency_matrices.png'))

    # 4. Node-link diagram
    plot_node_link_diagram(lines_data, all_graphs,
                           os.path.join(out_dir, 'node_link_diagram.png'))

    # 5. Edge statistics
    plot_edge_statistics(lines_data, all_graphs,
                         os.path.join(out_dir, 'edge_statistics.png'))

    # 6. Full dashboard
    plot_full_dashboard(original, lines_data, all_graphs, image_name,
                        os.path.join(out_dir, 'full_dashboard.png'))

    print(f"\n  All outputs saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate graph representations for GAT Nastaleeq OCR'
    )
    parser.add_argument('--image',  metavar='PATH',
                        help='Single image to process')
    parser.add_argument('--folder', metavar='PATH',
                        help='Process all images in a folder')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: output_grouped/<name>_graph_repr)')

    args = parser.parse_args()

    if args.image:
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_dir = args.outdir or os.path.join('output_grouped',
                                              f'{base}_graph_repr')
        process_image(args.image, out_dir)

    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"ERROR: folder not found: {args.folder}")
            sys.exit(1)
        for fname in sorted(os.listdir(args.folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(args.folder, fname)
                base = os.path.splitext(fname)[0]
                out_dir = args.outdir or os.path.join('output_grouped',
                                                      f'{base}_graph_repr')
                process_image(img_path, out_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python gat_graph_representation.py --image input_images/img-1.jpg")
        print("  python gat_graph_representation.py --folder input_images")


if __name__ == '__main__':
    main()

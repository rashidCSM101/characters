"""
GAT-Based Nastaleeq OCR Architecture
======================================
Implements the proposed methodology from:
  "A Generic Architecture for OCR of Arabic-Alphabet Based Languages
   Using Nastaleeq: A Graph Attention Approach"
  - Sayed Rashid Ali Shah, QAU

Pipeline:
    Image → CNN (ResNet-50) → Graph Construction → GAT (8 heads, 2 layers)
          → Bi-LSTM (2 layers) → CTC Decoder → Text

Dependencies:
    pip install torch torchvision torchaudio
    pip install torch-geometric  (optional, falls back to manual GAT)
    pip install opencv-python numpy

Usage:
    # Demo (graph construction only, no GPU needed):
    python gat_nastaleeq.py demo --image input_images/img3.png

    # Train:
    python gat_nastaleeq.py train --data_dir /path/to/upti --epochs 100

    # Inference:
    python gat_nastaleeq.py infer --image input_images/img3.png --model saved_model.pt
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import cv2
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Try importing PyTorch; if unavailable, provide helpful message and exit
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    _no_grad = torch.no_grad   # used as decorator below
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stub so class bodies that reference torch.no_grad parse correctly
    def _no_grad():
        """Stub: returns identity decorator when PyTorch is unavailable."""
        def decorator(fn):
            return fn
        return decorator

try:
    import torchvision
    from torchvision import models, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – PRE-PROCESSING  (Section 3.1 of paper)
#   Re-uses existing CompleteUrduOCR for grayscale, Otsu binarization, noise
#   removal, Hough skew-correction, horizontal-projection line segmentation,
#   and 8-connected-component analysis. This class wraps those steps so the
#   rest of the pipeline receives bounding boxes ready for graph construction.
# ══════════════════════════════════════════════════════════════════════════════

class NastaleeqPreprocessor:
    """
    Preprocessing stage: returns per-line lists of ligature bounding boxes.

    Steps (Section 3.1):
      1. Grayscale
      2. Gaussian blur σ=1.0 for noise suppression
      3. Morphological opening (3×3 kernel) to remove stray artifacts
      4. Otsu binarization
      5. Skew correction via Hough transform
      6. Horizontal projection profiling → line segmentation
      7. Per-line 8-connected component analysis → bounding boxes (x,y,w,h)
    """

    NUKTA_AREA_THRESHOLD = 150   # px² – from Table 1 (paper §3.3)

    def preprocess_image(self, image_path):
        """
        Load and binarize one image.
        Returns (binary_image, original_image).
        """
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        # 1. Grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian blur σ=1.0  (paper: "Gaussian filter σ=1.0 suppresses noise")
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)

        # 3. Morphological opening with 3×3 kernel  (paper: "remove stray artifacts")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

        # 4. Otsu binarization  (paper: "binarized using Otsu's method")
        _, binary = cv2.threshold(opened, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 5. Skew correction (paper: "Hough transform handles slight angular drift")
        binary = self._correct_skew(binary)

        return binary, original

    def segment_lines(self, binary):
        """
        Horizontal projection profiling → line boundary list.
        Returns list of (y_start, y_end) tuples.
        """
        h_proj = np.sum(binary, axis=1) / 255.0
        smoothed = np.convolve(h_proj, np.ones(5) / 5, mode='same')
        threshold = np.mean(smoothed) * 0.3

        lines = []
        in_line = False
        start = 0
        height = binary.shape[0]

        for i, val in enumerate(smoothed):
            if val > threshold and not in_line:
                start = i
                in_line = True
            elif val <= threshold and in_line:
                s = max(0, start - 5)
                e = min(height, i + 5)
                if e - s > 10:
                    lines.append((s, e))
                in_line = False

        if in_line:
            e = min(height, len(smoothed) + 5)
            if e - start > 10:
                lines.append((max(0, start - 5), e))

        return lines

    def extract_components(self, line_binary):
        """
        8-connected component analysis on one binary line image.
        Returns list of dicts with keys: x, y, w, h, area, cx, cy.
        (cx, cy) are bounding-box centroids, not moment centroids.
        """
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            line_binary, connectivity=8
        )
        components = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 3:       # discard single-pixel noise
                continue
            cx = x + w / 2.0  # bounding-box centroid x
            cy = y + h / 2.0  # bounding-box centroid y
            components.append(dict(x=x, y=y, w=w, h=h, area=area, cx=cx, cy=cy))
        return components

    def process(self, image_path):
        """
        Full preprocessing pipeline.

        Returns:
            lines_data: list of dicts, one per text line:
                {
                  'y_start': int, 'y_end': int,
                  'line_binary': ndarray,
                  'components': list of component dicts
                }
            binary: full binarized image
            original: original BGR image
        """
        binary, original = self.preprocess_image(image_path)
        line_bounds = self.segment_lines(binary)

        lines_data = []
        for y_start, y_end in line_bounds:
            line_bin = binary[y_start:y_end, :]
            comps = self.extract_components(line_bin)
            # Shift component y-coordinates into full-image space
            for c in comps:
                c['y_global'] = c['y'] + y_start
                c['cy_global'] = c['cy'] + y_start
            lines_data.append(dict(
                y_start=y_start, y_end=y_end,
                line_binary=line_bin,
                components=comps
            ))

        return lines_data, binary, original

    # ------------------------------------------------------------------
    def _correct_skew(self, binary):
        """
        Hough-transform-based skew correction (paper §3.1).
        """
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=80, minLineLength=80, maxLineGap=10)
        if lines is None:
            return binary

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)

        if not angles:
            return binary

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return binary

        h, w = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderValue=0)
        return rotated


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – GRAPH CONSTRUCTION  (Section 3.3 / Algorithm 1)
#   Builds G = (V, E, X) where each ligature is a node and edges encode the
#   three Nastaleeq-specific spatial relationships defined in the paper.
# ══════════════════════════════════════════════════════════════════════════════

EDGE_H = 0   # horizontal adjacency (reading order)
EDGE_V = 1   # vertical stacking
EDGE_N = 2   # nukta-to-base binding


class NastaleeqGraphConstructor:
    """
    Implements Algorithm 1 from the paper.

    Edge rules (Section 3.3):
      H (horizontal): |cy_i - cy_j| < 0.5 * max(h_i, h_j)
                      AND nodes are horizontally proximate
      V (vertical):   |cx_i - cx_j| < 0.5 * max(w_i, w_j)
                      AND |cy_i - cy_j| > 0.3 * max(h_i, h_j)
      N (nukta):      area(v_i) < τ=150 px²  →  v_i is nukta
                      connect to nearest non-nukta within r = 1.5 * h_nukta

    Node features (Eq. 2):
        x_i = f_i ⊕ [cx_i, cy_i, w_i, h_i]   → R^{256+4} = R^{260}
    where f_i is the CNN visual feature (provided later); for graph-only
    mode we use a zero vector of length 256.
    """

    NUKTA_AREA_THRESHOLD = 150   # τ  (Table 1)
    HORIZ_PROX_FACTOR    = 2.0   # maximum horizontal gap as multiple of avg width

    def build_graph(self, components, cnn_features=None):
        """
        Build the spatial graph for one text line.

        Args:
            components : list of component dicts (from NastaleeqPreprocessor)
            cnn_features: ndarray of shape (N, 256) or None.
                          If None, visual features are zeros.

        Returns dict with:
            node_features : ndarray (N, 260)
            edge_index    : ndarray (2, E) – source/target indices
            edge_type     : ndarray (E,)   – 0=H, 1=V, 2=N
            components    : echo of input (for downstream use)
        """
        n = len(components)
        if n == 0:
            return dict(node_features=np.zeros((0, 260), dtype=np.float32),
                        edge_index=np.zeros((2, 0), dtype=np.int64),
                        edge_type=np.zeros(0, dtype=np.int64),
                        components=[])

        if cnn_features is None:
            cnn_features = np.zeros((n, 256), dtype=np.float32)

        # ── Node features: f_i ⊕ [cx_i, cy_i, w_i, h_i]  (Eq. 2) ──────────
        spatial = np.array(
            [[c['cx'], c['cy'], c['w'], c['h']] for c in components],
            dtype=np.float32
        )
        node_features = np.concatenate([cnn_features, spatial], axis=1)  # (N, 260)

        # ── Edge construction (Algorithm 1, lines 7-15) ──────────────────────
        edges_src = []
        edges_dst = []
        edges_type = []

        # Identify nuktay (small dot components)
        is_nukta = [c['area'] < self.NUKTA_AREA_THRESHOLD for c in components]

        avg_w = np.mean([c['w'] for c in components]) if n > 0 else 1.0

        for i in range(n):
            ci = components[i]
            for j in range(n):
                if i == j:
                    continue
                cj = components[j]

                edge_t = self._classify_edge(ci, cj, is_nukta[i], is_nukta[j],
                                             avg_w)
                if edge_t is not None:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edges_type.append(edge_t)

        if not edges_src:
            # Isolated nodes: add self-loops to avoid empty adjacency
            for i in range(n):
                edges_src.append(i)
                edges_dst.append(i)
                edges_type.append(EDGE_H)

        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_type  = np.array(edges_type,              dtype=np.int64)

        return dict(node_features=node_features,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    components=components)

    # ------------------------------------------------------------------
    def _classify_edge(self, ci, cj, ci_is_nukta, cj_is_nukta, avg_w):
        """Return edge type (EDGE_H/V/N) or None if no edge."""
        # ── Nukta edge (N) – highest priority (Section 3.3) ───────────────
        if ci_is_nukta and not cj_is_nukta:
            r = 1.5 * ci['h']                  # search radius (Table 1)
            dist = math.hypot(ci['cx'] - cj['cx'], ci['cy'] - cj['cy'])
            if dist < r:
                return EDGE_N

        # ── Horizontal edge (H) ──────────────────────────────────────────
        # Condition: same vertical band AND horizontally proximate
        dy = abs(ci['cy'] - cj['cy'])
        vert_threshold = 0.5 * max(ci['h'], cj['h'])
        if dy < vert_threshold:
            # Also require some horizontal proximity (not across the whole line)
            dx = abs(ci['cx'] - cj['cx'])
            if dx < self.HORIZ_PROX_FACTOR * avg_w:
                return EDGE_H

        # ── Vertical edge (V) ────────────────────────────────────────────
        # Condition: horizontal overlap AND stacked vertically
        horiz_overlap = abs(ci['cx'] - cj['cx']) < 0.5 * max(ci['w'], cj['w'])
        vert_gap       = abs(ci['cy'] - cj['cy']) > 0.3 * max(ci['h'], cj['h'])
        if horiz_overlap and vert_gap:
            return EDGE_V

        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – CNN FEATURE EXTRACTOR  (Section 3.2)
#   ResNet-50 pretrained on ImageNet, fine-tuned on Urdu text.
#   Each 64×64 ligature crop → Linear(GAP(ResNet)) → f_i ∈ R^{256}
# ══════════════════════════════════════════════════════════════════════════════

def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the deep-learning components.\n"
            "Install with:  pip install torch torchvision"
        )


class LigatureCNNExtractor(nn.Module if TORCH_AVAILABLE else object):
    """
    Visual backbone: ResNet-50 → GAP → Linear(2048, 256).
    Input:  (B, 3, 64, 64) tensor
    Output: (B, 256) feature tensor                          (Eq. 1)
    """

    def __init__(self, pretrained=True, freeze_backbone=False):
        _require_torch()
        super().__init__()

        # ResNet-50 without the final classification head
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove avgpool and fc; we add our own projection
        self.feature_layers = nn.Sequential(*list(backbone.children())[:-2])
        # GAP → 2048-dim → 256-dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(2048, 256)

        if freeze_backbone:
            for p in self.feature_layers.parameters():
                p.requires_grad = False

    def forward(self, x):
        """x: (B, 3, 64, 64)  →  (B, 256)"""
        feat = self.feature_layers(x)   # (B, 2048, 2, 2)
        feat = self.gap(feat)            # (B, 2048, 1, 1)
        feat = feat.flatten(1)           # (B, 2048)
        feat = self.projection(feat)     # (B, 256)
        return feat

    # ------------------------------------------------------------------
    @staticmethod
    def crop_ligatures(line_binary, components, crop_size=64):
        """
        Helper: crop each ligature region from a binary line image,
        convert to 3-channel float32 tensor normalised to [0,1].

        Returns tensor of shape (N, 3, crop_size, crop_size).
        """
        _require_torch()
        crops = []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),                          # [0,1]
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),   # ImageNet stats
        ])
        for c in components:
            x, y, w, h = int(c['x']), int(c['y']), int(c['w']), int(c['h'])
            patch = line_binary[y:y+h, x:x+w]
            if patch.size == 0:
                patch = np.zeros((crop_size, crop_size), dtype=np.uint8)
            # Binary to 3-channel RGB
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            crops.append(transform(patch_rgb))
        if not crops:
            return torch.zeros(0, 3, crop_size, crop_size)
        return torch.stack(crops)                           # (N, 3, 64, 64)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – GRAPH ATTENTION NETWORK  (Section 3.4)
#   Manual GAT implementation (no torch_geometric dependency).
#
#   Equations from paper:
#     e_ij  = LeakyReLU(a^T [Wh_i || Wh_j])            Eq.3
#     α_ij  = softmax_j(e_ij)                            Eq.4
#     h'_i  = σ(Σ_j α_ij W h_j)                         Eq.5
#     h''_i = ||_{k=1}^{K} σ(Σ_j α^k_ij W^k h_j)       Eq.6
#
#   Hyperparams (Table 1):
#     K = 8 heads, hidden dim 256 per head (query/key projection)
#     output per head = 64 → total output per layer = 8×64 = 512
#     2 stacked GAT layers → h''_i ∈ R^{512}
# ══════════════════════════════════════════════════════════════════════════════

class GATLayer(nn.Module if TORCH_AVAILABLE else object):
    """
    Single Graph Attention layer with K multi-heads.

    Args:
        in_dim    : input feature dimension
        out_dim   : output dimension per head
        num_heads : K (default 8)
        dropout   : attention dropout probability
        concat    : if True,  output is (N, K*out_dim)
                    if False, output is (N, out_dim) [mean over heads]
    """

    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.0, concat=True):
        _require_torch()
        super().__init__()

        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.num_heads = num_heads
        self.concat    = concat

        # One W matrix per head: (K, in_dim, out_dim)
        self.W = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        # Attention vector per head: (K, 2*out_dim)
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout    = nn.Dropout(p=dropout)
        self.activation = nn.ELU()

        nn.init.xavier_uniform_(self.W.view(num_heads * in_dim, out_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))

    def forward(self, h, edge_index):
        """
        Args:
            h          : (N, in_dim) node features
            edge_index : (2, E)  long tensor – [source_row, target_row]

        Returns:
            h_out : (N, K*out_dim) if concat else (N, out_dim)
        """
        N = h.size(0)
        K = self.num_heads

        # Project all nodes: (K, N, out_dim)
        Wh = torch.einsum('kio,ni->kno', self.W, h)   # (K, N, out_dim)

        src, dst = edge_index[0], edge_index[1]        # (E,)

        # Gather features for each edge endpoint: (K, E, out_dim)
        Wh_src = Wh[:, src, :]   # (K, E, out_dim)
        Wh_dst = Wh[:, dst, :]   # (K, E, out_dim)

        # Attention scores: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        concat_feats = torch.cat([Wh_src, Wh_dst], dim=-1)          # (K, E, 2*out_dim)
        e = self.leaky_relu(
            torch.einsum('keo,ko->ke', concat_feats, self.a)         # (K, E)
        )

        # Softmax normalisation per destination node (Eq. 4)
        # For each dst node, normalise across all its incoming neighbours
        alpha = self._sparse_softmax(e, dst, N)   # (K, E)
        alpha = self.dropout(alpha)

        # Aggregate attended neighbour features (Eq. 5)
        # h'_i = σ(Σ_j α_ij Wh_j)
        # For each head k: scatter_add over dst
        agg = torch.zeros(K, N, self.out_dim, device=h.device)
        for k in range(K):
            weighted = alpha[k].unsqueeze(-1) * Wh[k, src, :]   # (E, out_dim)
            agg[k].scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)

        agg = self.activation(agg)   # (K, N, out_dim)

        if self.concat:
            # Concat heads: (N, K*out_dim)
            return agg.permute(1, 0, 2).reshape(N, K * self.out_dim)
        else:
            # Mean over heads: (N, out_dim)
            return agg.mean(dim=0)

    @staticmethod
    def _sparse_softmax(scores, idx, num_nodes):
        """
        Numerically stable per-node softmax over edge scores.
        scores : (K, E)
        idx    : (E,) – destination nodes
        Returns: (K, E) normalised attention weights
        """
        K, E = scores.shape
        # Subtract max per destination (stability)
        max_scores = torch.full((K, num_nodes), float('-inf'),
                                device=scores.device)
        max_scores.scatter_reduce_(1, idx.unsqueeze(0).expand(K, E),
                                   scores, reduce='amax', include_self=True)
        scores_shifted = scores - max_scores[:, idx]   # (K, E)

        exp_scores = scores_shifted.exp()

        # Sum of exp per destination
        sum_exp = torch.zeros(K, num_nodes, device=scores.device)
        sum_exp.scatter_add_(1, idx.unsqueeze(0).expand(K, E), exp_scores)

        # Avoid division by zero
        sum_exp = sum_exp.clamp(min=1e-12)
        alpha = exp_scores / sum_exp[:, idx]   # (K, E)
        return alpha


class NastaleeqGATModule(nn.Module if TORCH_AVAILABLE else object):
    """
    Two stacked GAT layers (Section 3.4):
      Layer 1: (260) → (512)    [8 heads × 64 per head]
      Layer 2: (512) → (512)    [8 heads × 64 per head]

    Final output h''_i ∈ R^{512} as specified in the paper.
    """

    GAT_HEADS     = 8
    HEAD_OUT_DIM  = 64    # per head; 8×64 = 512 total  (Table 1: 2 layers)
    NODE_IN_DIM   = 260   # CNN(256) + spatial(4)

    def __init__(self, dropout=0.1):
        _require_torch()
        super().__init__()

        inter_dim = self.GAT_HEADS * self.HEAD_OUT_DIM   # 512

        self.gat1 = GATLayer(self.NODE_IN_DIM, self.HEAD_OUT_DIM,
                             num_heads=self.GAT_HEADS, dropout=dropout, concat=True)
        self.gat2 = GATLayer(inter_dim, self.HEAD_OUT_DIM,
                             num_heads=self.GAT_HEADS, dropout=dropout, concat=True)

        self.norm1 = nn.LayerNorm(inter_dim)
        self.norm2 = nn.LayerNorm(inter_dim)

    def forward(self, node_features, edge_index):
        """
        Args:
            node_features : (N, 260) float tensor
            edge_index    : (2, E)   long tensor

        Returns:
            h : (N, 512) spatially-enriched embeddings
        """
        # Layer 1
        h = self.gat1(node_features, edge_index)   # (N, 512)
        h = self.norm1(h)

        # Layer 2 (wider contextual aggregation)
        h = self.gat2(h, edge_index)               # (N, 512)
        h = self.norm2(h)

        return h   # h''_i ∈ R^{512}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – BI-LSTM DECODER  (Section 3.5 & 3.6)
#   After GAT, nodes are sorted RTL by x-centroid (Eq. 7), then fed to a
#   2-layer bidirectional LSTM with 256 units per direction (512 per step).
#   CTC loss (Eq. 8) with beam search decoding (beam width 10).
# ══════════════════════════════════════════════════════════════════════════════

class NastaleeqOCRModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Full proposed architecture (Figure 1):
      CNN → Graph + GAT → Bi-LSTM → CTC

    Args:
        vocab_size     : |C| – number of output characters + 1 blank
        pretrained_cnn : whether to initialise ResNet-50 with ImageNet weights
        freeze_cnn     : whether to freeze CNN during training
    """

    LSTM_HIDDEN   = 256    # units per direction (Table 1)
    LSTM_LAYERS   = 2      # Table 1
    LSTM_DROPOUT  = 0.3    # Table 1
    GAT_DIM       = 512    # h''_i dimension

    def __init__(self, vocab_size, pretrained_cnn=True, freeze_cnn=False):
        _require_torch()
        super().__init__()

        self.cnn = LigatureCNNExtractor(pretrained=pretrained_cnn,
                                        freeze_backbone=freeze_cnn)
        self.gat = NastaleeqGATModule()

        # Sequence decoder (Section 3.5)
        self.bilstm = nn.LSTM(
            input_size=self.GAT_DIM,
            hidden_size=self.LSTM_HIDDEN,
            num_layers=self.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=self.LSTM_DROPOUT if self.LSTM_LAYERS > 1 else 0.0
        )
        # Projection to character vocabulary (+ CTC blank)
        self.fc = nn.Linear(self.LSTM_HIDDEN * 2, vocab_size)

    def forward(self, ligature_crops, edge_index, cx_order):
        """
        Forward pass for one text line.

        Args:
            ligature_crops : (N, 3, 64, 64) – pre-cropped ligature images
            edge_index     : (2, E) long tensor – spatial graph edges
            cx_order       : (N,) long tensor – indices that sort nodes RTL
                             i.e. argsort(−cx)  →  π in Eq.7

        Returns:
            log_probs : (N_seq, vocab_size) log-softmax output for CTC
        """
        # ── CNN feature extraction (Section 3.2, Eq.1) ───────────────────
        cnn_feats = self.cnn(ligature_crops)   # (N, 256)

        # ── Append spatial coordinates to get 260-dim node features ──────
        # NOTE: spatial coordinates are embedded in the graph data;
        #       here we concatenate a simple positional encoding derived
        #       from cx_order position (0..N-1 normalised) as proxy.
        #       During full training, pass actual [cx,cy,w,h] from graph.
        N = cnn_feats.size(0)
        spatial = torch.zeros(N, 4, device=cnn_feats.device)
        if hasattr(self, '_spatial_cache'):
            spatial = self._spatial_cache
        node_feats = torch.cat([cnn_feats, spatial], dim=1)  # (N, 260)

        # ── GAT spatial bridge (Section 3.4, Eqs.3-6) ───────────────────
        enriched = self.gat(node_feats, edge_index)   # (N, 512)

        # ── Sequence ordering: sort RTL by x-centroid (Eq.7) ────────────
        ordered = enriched[cx_order]                   # (N, 512)

        # ── Bi-LSTM (Section 3.5) ────────────────────────────────────────
        seq = ordered.unsqueeze(0)                     # (1, N, 512)
        lstm_out, _ = self.bilstm(seq)                 # (1, N, 512)
        lstm_out = lstm_out.squeeze(0)                 # (N, 512)

        # ── CTC projection ───────────────────────────────────────────────
        logits    = self.fc(lstm_out)                  # (N, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)      # (N, vocab_size)

        return log_probs

    def set_spatial_features(self, spatial_tensor):
        """Inject [cx,cy,w,h] spatial features before forward()."""
        self._spatial_cache = spatial_tensor


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – DATASET  (Section 4.1)
#   Wraps UPTI-format data: each sample is a text-line image + Unicode label.
# ══════════════════════════════════════════════════════════════════════════════

class UPTIDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Lightweight dataset for UPTI (Urdu Printed Text Images) format.

    Expected directory structure:
        data_dir/
          images/  img_0001.png  img_0002.png ...
          labels/  img_0001.txt  img_0002.txt ...

    Each .txt file contains the Unicode ground-truth transcription.
    """

    def __init__(self, data_dir, vocab, split='train', split_ratio=(0.8, 0.1, 0.1)):
        _require_torch()
        self.data_dir    = data_dir
        self.vocab       = vocab        # NastaleeqVocab instance
        self.preprocessor = NastaleeqPreprocessor()
        self.graph_ctor   = NastaleeqGraphConstructor()

        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')

        all_samples = []
        if os.path.isdir(images_dir):
            for fname in sorted(os.listdir(images_dir)):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(images_dir, fname)
                lbl_path = os.path.join(labels_dir,
                                        os.path.splitext(fname)[0] + '.txt')
                if os.path.exists(lbl_path):
                    with open(lbl_path, encoding='utf-8') as f:
                        label = f.read().strip()
                    all_samples.append((img_path, label))

        # Deterministic train/val/test split
        n = len(all_samples)
        t_end = int(n * split_ratio[0])
        v_end = t_end + int(n * split_ratio[1])
        if split == 'train':
            self.samples = all_samples[:t_end]
        elif split == 'val':
            self.samples = all_samples[t_end:v_end]
        else:
            self.samples = all_samples[v_end:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        lines_data, _, _ = self.preprocessor.process(img_path)

        # Use first detected text line (single-line UPTI images)
        if not lines_data:
            comps = []
            line_bin = np.zeros((64, 256), dtype=np.uint8)
        else:
            ld      = lines_data[0]
            comps   = ld['components']
            line_bin = ld['line_binary']

        graph = self.graph_ctor.build_graph(comps)

        crops    = LigatureCNNExtractor.crop_ligatures(line_bin, comps)
        target   = torch.tensor(self.vocab.encode(label), dtype=torch.long)
        cx_order = self._rtl_order(comps)

        # Spatial features [cx, cy, w, h] for each node
        if comps:
            spatial = torch.tensor(
                [[c['cx'], c['cy'], c['w'], c['h']] for c in comps],
                dtype=torch.float32
            )
        else:
            spatial = torch.zeros(0, 4)

        return dict(
            crops=crops,
            edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
            edge_type=torch.tensor(graph['edge_type'],   dtype=torch.long),
            cx_order=cx_order,
            spatial=spatial,
            target=target,
            target_len=torch.tensor(len(target), dtype=torch.long),
            input_len=torch.tensor(max(len(comps), 1), dtype=torch.long),
        )

    @staticmethod
    def _rtl_order(components):
        """Return indices that sort components right-to-left (Eq. 7)."""
        if not components:
            return torch.zeros(0, dtype=torch.long)
        cx_vals = np.array([c['cx'] for c in components])
        return torch.tensor(np.argsort(-cx_vals), dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – VOCABULARY
# ══════════════════════════════════════════════════════════════════════════════

class NastaleeqVocab:
    """
    Minimal character vocabulary for Urdu Nastaleeq.
    CTC blank index = 0.
    """

    # Core Urdu Unicode characters (extend as needed for your dataset)
    URDU_CHARS = (
        "\u0627\u0628\u067e\u062a\u0679\u062b\u062c\u0686\u062d"
        "\u062e\u062f\u0688\u0630\u0631\u0691\u0632\u0698\u0633"
        "\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0641\u0642"
        "\u06a9\u06af\u0644\u0645\u0646\u0648\u06be\u06cc\u06d2"
        "\u0621\u0622\u0623\u0626\u06c1\u06c3"
        "\u0020"  # space
    )

    def __init__(self, extra_chars=""):
        blank = ["\x00"]   # CTC blank at index 0
        chars = blank + list(dict.fromkeys(self.URDU_CHARS + extra_chars))
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.blank_idx = 0

    def __len__(self):
        return len(self.idx2char)

    def encode(self, text):
        return [self.char2idx.get(c, self.blank_idx) for c in text]

    def decode_greedy(self, log_probs_tensor):
        """CTC greedy decode – collapse repeats and remove blanks."""
        indices = log_probs_tensor.argmax(-1).tolist()
        result  = []
        prev    = None
        for idx in indices:
            if idx != prev and idx != self.blank_idx:
                result.append(self.idx2char.get(idx, '?'))
            prev = idx
        return ''.join(result)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – TRAINER  (Section 4 – Evaluation Plan)
#   Implements Adam optimiser, cosine LR decay, early stopping, CTC loss.
# ══════════════════════════════════════════════════════════════════════════════

class NastaleeqTrainer:
    """
    Training loop for NastaleeqOCRModel.

    Hyperparameters follow Table 1:
      lr=1e-4, cosine decay, batch=32, epochs=100, early_stopping patience=10
    """

    def __init__(self, model, vocab, device='cpu'):
        _require_torch()
        self.model  = model.to(device)
        self.vocab  = vocab
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999)
        )
        self.ctc_loss = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            # Single-sample batches (variable graph sizes; collate_fn not set)
            crops      = batch['crops'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            cx_order   = batch['cx_order'].to(self.device)
            spatial    = batch['spatial'].to(self.device)
            target     = batch['target'].to(self.device)
            target_len = batch['target_len'].to(self.device)
            input_len  = batch['input_len'].to(self.device)

            if crops.size(0) == 0:
                continue

            self.model.set_spatial_features(spatial)
            self.optimizer.zero_grad()

            log_probs = self.model(crops, edge_index, cx_order)
            # CTC expects (T, N, C) – here T=seq_len, N=1 (single sample)
            lp = log_probs.unsqueeze(1)   # (T, 1, C)

            loss = self.ctc_loss(lp, target.unsqueeze(0),
                                 input_len.unsqueeze(0), target_len.unsqueeze(0))
            if not loss.isnan():
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    @_no_grad()
    def evaluate(self, dataloader):
        """Returns average CTC loss on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            crops      = batch['crops'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            cx_order   = batch['cx_order'].to(self.device)
            spatial    = batch['spatial'].to(self.device)
            target     = batch['target'].to(self.device)
            target_len = batch['target_len'].to(self.device)
            input_len  = batch['input_len'].to(self.device)

            if crops.size(0) == 0:
                continue

            self.model.set_spatial_features(spatial)
            log_probs = self.model(crops, edge_index, cx_order)
            lp = log_probs.unsqueeze(1)
            loss = self.ctc_loss(lp, target.unsqueeze(0),
                                 input_len.unsqueeze(0), target_len.unsqueeze(0))
            if not loss.isnan():
                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def train(self, train_loader, val_loader, epochs=100,
              patience=10, scheduler_t_max=100, save_path='nastaleeq_gat.pt'):
        """
        Full training loop with cosine LR decay and early stopping.
        """
        scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=scheduler_t_max
        )
        best_val     = float('inf')
        no_improve   = 0

        print("="*65)
        print("NastaleeqGAT Training")
        print(f"  Epochs={epochs}  patience={patience}  device={self.device}")
        print("="*65)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.evaluate(val_loader)
            scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  [ok] Model saved (val={best_val:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        print(f"\n[ok] Training complete. Best val loss: {best_val:.4f}")
        print(f"[ok] Model saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – GRAPH VISUALISER  (demo-mode, no PyTorch needed)
#   Draws the three typed edges on the line image for inspection.
# ══════════════════════════════════════════════════════════════════════════════

EDGE_COLOURS = {
    EDGE_H: (0,   165, 255),   # orange  – horizontal (reading order)
    EDGE_V: (255,   0, 255),   # magenta – vertical   (stacking)
    EDGE_N: (0,   255,   0),   # green   – nukta-to-base
}
EDGE_LABELS = {EDGE_H: 'H', EDGE_V: 'V', EDGE_N: 'N'}


def visualise_graph(line_bgr, components, graph_data, out_path=None):
    """
    Draw component bounding boxes and coloured typed edges on a BGR image.

    Args:
        line_bgr    : (H, W, 3) or (H, W) image
        components  : list of component dicts
        graph_data  : dict returned by NastaleeqGraphConstructor.build_graph()
        out_path    : if given, save visualisation to this path

    Returns:
        vis : annotated BGR image
    """
    if len(line_bgr.shape) == 2:
        vis = cv2.cvtColor(line_bgr, cv2.COLOR_GRAY2BGR)
    else:
        vis = line_bgr.copy()

    edge_index = graph_data['edge_index']   # (2, E)
    edge_type  = graph_data['edge_type']    # (E,)
    nukta_thr  = NastaleeqGraphConstructor.NUKTA_AREA_THRESHOLD

    # Draw bounding boxes
    for c in components:
        is_nukta = c['area'] < nukta_thr
        colour  = (0, 255, 255) if is_nukta else (255, 255, 0)
        cv2.rectangle(vis,
                      (c['x'], c['y']),
                      (c['x'] + c['w'], c['y'] + c['h']),
                      colour, 1)

    # Draw typed edges
    for k in range(edge_index.shape[1]):
        i  = int(edge_index[0, k])
        j  = int(edge_index[1, k])
        et = int(edge_type[k])
        c1 = (int(components[i]['cx']), int(components[i]['cy']))
        c2 = (int(components[j]['cx']), int(components[j]['cy']))
        cv2.arrowedLine(vis, c1, c2, EDGE_COLOURS.get(et, (128, 128, 128)),
                        1, tipLength=0.15)
        mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
        cv2.putText(vis, EDGE_LABELS.get(et, '?'), mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    EDGE_COLOURS.get(et, (128, 128, 128)), 1)

    # Legend
    legend_y = 12
    for et, label in EDGE_LABELS.items():
        cv2.putText(vis, f"{label}: {['horizontal','vertical','nukta'][et]}",
                    (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    EDGE_COLOURS[et], 1)
        legend_y += 14

    if out_path:
        cv2.imwrite(out_path, vis)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class NastaleeqInferencePipeline:
    """
    End-to-end inference: image path → predicted Urdu text.

    Wraps all stages: preprocessing → graph + GAT → Bi-LSTM → CTC decode.
    """

    def __init__(self, model_path, vocab, device='cpu'):
        _require_torch()
        self.vocab       = vocab
        self.device      = device
        self.preprocessor = NastaleeqPreprocessor()
        self.graph_ctor   = NastaleeqGraphConstructor()
        self.model        = NastaleeqOCRModel(vocab_size=len(vocab)).to(device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.eval()

    @_no_grad()
    def predict(self, image_path):
        """
        Predict text for all lines in an image.
        Returns list of predicted strings, one per detected text line.
        """
        lines_data, _, _ = self.preprocessor.process(image_path)
        results = []

        for ld in lines_data:
            comps    = ld['components']
            line_bin = ld['line_binary']

            if not comps:
                results.append('')
                continue

            graph    = self.graph_ctor.build_graph(comps)
            crops    = LigatureCNNExtractor.crop_ligatures(line_bin, comps)
            crops    = crops.to(self.device)

            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long,
                                      device=self.device)
            cx_order   = UPTIDataset._rtl_order(comps).to(self.device)
            spatial    = torch.tensor(
                [[c['cx'], c['cy'], c['w'], c['h']] for c in comps],
                dtype=torch.float32, device=self.device
            )

            self.model.set_spatial_features(spatial)
            log_probs = self.model(crops, edge_index, cx_order)
            text      = self.vocab.decode_greedy(log_probs)
            results.append(text)

        return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – GRAPH STATS REPORTER (no PyTorch needed)
#   Prints edge-type distribution for a processed image.
# ══════════════════════════════════════════════════════════════════════════════

def report_graph_stats(lines_data):
    """Print graph statistics for all lines in a processed image."""
    gc = NastaleeqGraphConstructor()
    total_nodes = total_H = total_V = total_N = 0

    for idx, ld in enumerate(lines_data, 1):
        comps = ld['components']
        g     = gc.build_graph(comps)
        et    = g['edge_type']
        n_H   = int(np.sum(et == EDGE_H))
        n_V   = int(np.sum(et == EDGE_V))
        n_N   = int(np.sum(et == EDGE_N))

        print(f"  Line {idx:2d}: {len(comps):3d} nodes | "
              f"H={n_H:3d}  V={n_V:3d}  N={n_N:3d}  "
              f"total_edges={len(et):3d}")

        total_nodes += len(comps)
        total_H += n_H;  total_V += n_V;  total_N += n_N

    print(f"\n  TOTALS: {total_nodes} nodes | "
          f"H={total_H}  V={total_V}  N={total_N}  "
          f"total_edges={total_H+total_V+total_N}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def cmd_demo(args):
    """
    Demo mode: run preprocessing + graph construction + visualisation.
    Does NOT require PyTorch.
    """
    print("="*65)
    print("NastaleeqGAT - Graph Construction Demo")
    print("="*65)

    prep = NastaleeqPreprocessor()
    gc   = NastaleeqGraphConstructor()

    print(f"\nProcessing: {args.image}")
    lines_data, binary, original = prep.process(args.image)
    print(f"  -> {len(lines_data)} text line(s) detected")

    report_graph_stats(lines_data)

    # Save per-line graph visualisations
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join("output_grouped", f"{base}_gat_graphs")
    os.makedirs(out_dir, exist_ok=True)

    for idx, ld in enumerate(lines_data, 1):
        comps    = ld['components']
        line_bin = ld['line_binary']
        g        = gc.build_graph(comps)

        line_bgr = cv2.cvtColor(line_bin, cv2.COLOR_GRAY2BGR)
        out_path = os.path.join(out_dir, f"line_{idx:02d}_graph.png")
        visualise_graph(line_bgr, comps, g, out_path=out_path)
        print(f"  [ok] Graph saved: {out_path}")

    # Save graph data as JSON for inspection
    json_out = []
    for idx, ld in enumerate(lines_data, 1):
        comps = ld['components']
        g     = gc.build_graph(comps)
        et    = g['edge_type'].tolist()
        ei    = g['edge_index'].tolist()
        json_out.append(dict(
            line=idx,
            num_nodes=len(comps),
            edges=[dict(src=ei[0][k], dst=ei[1][k],
                        type=EDGE_LABELS[et[k]])
                   for k in range(len(et))]
        ))

    json_path = os.path.join(out_dir, "graph_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n  [ok] Graph JSON: {json_path}")
    print(f"\n  Edge legend: H=horizontal  V=vertical  N=nukta-to-base")
    print("="*65)


def cmd_train(args):
    """Training mode."""
    _require_torch()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    vocab = NastaleeqVocab()
    model = NastaleeqOCRModel(vocab_size=len(vocab),
                              pretrained_cnn=True, freeze_cnn=False)

    train_ds = UPTIDataset(args.data_dir, vocab, split='train')
    val_ds   = UPTIDataset(args.data_dir, vocab, split='val')
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    # Single-sample DataLoader (graphs vary per line – no batching without
    # custom collate; extend with pad/pack for batch training)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    trainer = NastaleeqTrainer(model, vocab, device=device)
    trainer.train(train_loader, val_loader,
                  epochs=args.epochs,
                  patience=args.patience,
                  save_path=args.save)


def cmd_infer(args):
    """Inference mode: predict text in a single image."""
    _require_torch()
    vocab    = NastaleeqVocab()
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = NastaleeqInferencePipeline(args.model, vocab, device=device)
    print(f"Predicting: {args.image}")
    predictions = pipeline.predict(args.image)
    for i, pred in enumerate(predictions, 1):
        print(f"  Line {i}: {pred if pred else '[empty]'}")


def main():
    parser = argparse.ArgumentParser(
        description="NastaleeqGAT – Graph Attention OCR for Urdu/Nastaleeq"
    )
    sub = parser.add_subparsers(dest='cmd')

    # ── demo ──
    p_demo = sub.add_parser('demo', help='Graph construction demo (no GPU)')
    p_demo.add_argument('--image', required=True, help='Input image path')

    # ── train ──
    p_train = sub.add_parser('train', help='Train the full model')
    p_train.add_argument('--data_dir', required=True,
                         help='UPTI dataset root (images/ + labels/)')
    p_train.add_argument('--epochs',  type=int, default=100)
    p_train.add_argument('--patience',type=int, default=10)
    p_train.add_argument('--save',    default='nastaleeq_gat.pt')

    # ── infer ──
    p_inf = sub.add_parser('infer', help='Run inference on an image')
    p_inf.add_argument('--image', required=True)
    p_inf.add_argument('--model', required=True, help='Path to saved .pt model')

    args = parser.parse_args()

    if args.cmd == 'demo':
        cmd_demo(args)
    elif args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'infer':
        cmd_infer(args)
    else:
        parser.print_help()
        print("\nQuick start (no GPU required):")
        print("  python gat_nastaleeq.py demo --image input_images/img3.png")
        print("\nFull training (requires PyTorch + UPTI dataset):")
        print("  python gat_nastaleeq.py train --data_dir /path/to/upti")


if __name__ == '__main__':
    main()

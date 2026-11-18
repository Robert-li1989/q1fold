# Adapted from sincFold (https://github.com/sinc-lab/sincFold), GPL-3.0.
# Modifications for Q1Fold (c) 2026 Zeyu Li et al., GPL-3.0.
"""
Q1Fold Metrics Module

Provides evaluation metrics for RNA secondary structure prediction.
"""

import torch as tr
from sklearn.metrics import f1_score
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, ShortestPath
import numpy as np 

from .utils import mat2bp


def contact_f1(ref_batch, pred_batch, L, th=0.5, reduce=True, method="triangular", return_all_metrics=False):
    """
    Compute F1 from base pairs. Input goes to sigmoid and then thresholded.
    
    Args:
        ref_batch: Reference contact matrices [B, N, N]
        pred_batch: Predicted contact matrices [B, N, N]
        L: List of sequence lengths
        th: Threshold for binarization
        reduce: Whether to average over batch
        method: 'triangular' or 'shift'
        return_all_metrics: If True, also return precision and recall
        
    Returns:
        F1 score (and optionally precision, recall)
    """
    f1_list = []
    precision_list = []
    recall_list = []

    if type(ref_batch) == float or len(ref_batch.shape) < 3:
        ref_batch = [ref_batch]
        pred_batch = [pred_batch]
        L = [L]

    for ref, pred, l in zip(ref_batch, pred_batch, L):
        # ignore padding
        ind = tr.where(ref != -1)
        pred = pred[ind].view(l, l)
        ref = ref[ind].view(l, l)
        
        # pred goes from -inf to inf
        pred = tr.sigmoid(pred)
        pred[pred <= th] = 0

        if method == "triangular":
            f1 = f1_triangular(ref, pred > th)
            # Calculate precision and recall for triangular method
            if return_all_metrics:
                # get upper triangular matrix without diagonal
                ind = tr.triu_indices(ref.shape[0], ref.shape[1], offset=1)
                ref_flat = ref[ind[0], ind[1]].numpy().ravel()
                pred_flat = (pred > th)[ind[0], ind[1]].numpy().ravel()

                tp = np.logical_and(pred_flat, ref_flat).sum()
                fp = pred_flat.sum() - tp
                fn = ref_flat.sum() - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                precision_list.append(precision)
                recall_list.append(recall)

        elif method == "shift":
            ref = mat2bp(ref)
            pred = mat2bp(pred)
            recall, precision, f1 = f1_shift(ref, pred)
            if return_all_metrics:
                precision_list.append(precision)
                recall_list.append(recall)
        else:
            raise NotImplementedError

        f1_list.append(f1)

    if return_all_metrics:
        if reduce:
            return (
                tr.tensor(f1_list).mean().item(),
                tr.tensor(precision_list).mean().item() if precision_list else 0.0,
                tr.tensor(recall_list).mean().item() if recall_list else 0.0
            )
        else:
            return tr.tensor(f1_list), tr.tensor(precision_list), tr.tensor(recall_list)
    else:
        if reduce:
            return tr.tensor(f1_list).mean().item()
        else:
            return tr.tensor(f1_list)


def f1_triangular(ref, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = tr.triu_indices(ref.shape[0], ref.shape[1], offset=1)

    ref = ref[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(ref, pred, zero_division=0)


def f1_shift(ref_bp, pre_bp):
    """F1 score with tolerance of 1 position"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if (
            rbp in pre_bp
            or [rbp[0], rbp[1] - 1] in pre_bp
            or [rbp[0], rbp[1] + 1] in pre_bp
            or [rbp[0] + 1, rbp[1]] in pre_bp
            or [rbp[0] - 1, rbp[1]] in pre_bp
        ):
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if (
            pbp in ref_bp
            or [pbp[0], pbp[1] - 1] in ref_bp
            or [pbp[0], pbp[1] + 1] in ref_bp
            or [pbp[0] + 1, pbp[1]] in ref_bp
            or [pbp[0] - 1, pbp[1]] in ref_bp
        ):
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1


def f1_strict(ref_bp, pre_bp):
    """F1 score strict, same as triangular but less efficient"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if rbp in pre_bp:
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if pbp in ref_bp:
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1


def mcc(pred, ref):
    """Matthews correlation coefficient. pred, ref are 2D binary matrices"""
    tp = np.logical_and(pred, ref).sum()
    fp = pred.sum() - tp
    fn = ref.sum() - tp
    tn = np.logical_and(np.logical_not(pred), np.logical_not(ref)).sum()
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    return mcc


def get_graph_kernel(kernel, n_iter=5, normalize=True):
    """Get graph kernel for structure comparison."""
    if kernel == 'WeisfeilerLehman':
        return WeisfeilerLehman(
            n_iter=n_iter,
            normalize=normalize,
            base_graph_kernel=VertexHistogram
        )
    elif kernel == 'WeisfeilerLehmanOptimalAssignment':
        return WeisfeilerLehmanOptimalAssignment(
            n_iter=n_iter,
            normalize=normalize
        )
    elif kernel == 'ShortestPath':
        return ShortestPath(normalize=normalize)
    

def mat2graph(matrix, node_labels=None):
    """Convert contact matrix to graph."""
    if node_labels is not None:
        node_labels = {s: str(s) for s in range(matrix.shape[0])}
    graph = Graph(initialization_object=matrix.astype(int), node_labels=node_labels)  
    return graph


def WL(pred, ref, sequence=None, kernel="WeisfeilerLehman", n_iter=5):
    """Weisfeiler-Lehman graph kernel similarity. pred, ref are 2D binary matrices"""
    
    node_labels = None
    if sequence is not None:
        node_labels = {i: s for i, s in enumerate(sequence)}
    
    pred_graph = mat2graph(pred, node_labels=node_labels)
    true_graph = mat2graph(ref, node_labels=node_labels)
    kernel = get_graph_kernel(kernel=kernel, n_iter=n_iter)

    kernel.fit_transform([true_graph])
    distance_score = kernel.transform([pred_graph])  

    return distance_score[0][0]

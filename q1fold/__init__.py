# Adapted from sincFold (https://github.com/sinc-lab/sincFold), GPL-3.0.
# Modifications for Q1Fold (c) 2026 Zeyu Li et al., GPL-3.0.
"""
Q1Fold: A Hybrid Quantum-Classical CNN for RNA Secondary Structure Prediction

Q1Fold integrates a 6-qubit variational quantum circuit with a classical 2D ResNet
architecture for RNA secondary structure prediction. The quantum circuit generates
expressive features from local sequence windows using minimal qubits.

Usage:
    # Command line
    python -m q1fold train data.csv -o results/
    python -m q1fold pred sequence.fasta -o predictions.csv
    
    # Python API
    from q1fold import q1fold
    model = q1fold(weights='path/to/weights.pmt')
    predictions, _ = model.pred(data_loader)
"""

# Suppress torch.load FutureWarning (from PennyLane internals)
import warnings
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*verbose parameter is deprecated.*", category=UserWarning)

import json
import os
import random
import numpy as np
import torch as tr
from datetime import datetime
import pandas as pd
import shutil
import pickle

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from .dataset import SeqDataset, pad_batch
from .model import q1fold, Q1Fold
from .embeddings import NT_DICT, VOCABULARY, OneHotEmbedding
from .utils import write_ct, validate_file, ct2dot, dot2png, ct2svg
from .parser import parser
from ._version import __version__


# Expose main classes and functions
__all__ = [
    'q1fold',
    'Q1Fold', 
    'SeqDataset',
    'pad_batch',
    'main',
    '__version__',
]


def main():
    """Main entry point for command line interface."""
    
    args = parser()
    
    if not args.no_cache and args.command == "train":
        cache_path = "cache/"
    else:
        cache_path = None

    config = {
        "device": args.d, 
        "batch_size": args.batch, 
        "valid_split": 0.1, 
        "max_len": args.max_length, 
        "verbose": not args.quiet, 
        "cache_path": cache_path
    }
    
    if "max_epochs" in args:
        config["max_epochs"] = args.max_epochs

    if args.config is not None:
        config.update(json.load(open(args.config)))

    if config["cache_path"] is not None:
        shutil.rmtree(config["cache_path"], ignore_errors=True)
        os.makedirs(config["cache_path"])

    # Reproducibility
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if args.command == "train": 
        if hasattr(args, 'folds') and args.folds is not None:
            train_kfold(args.train_file, config, args.out_path, args.folds, args.j)
        else:
            train(args.train_file, config, args.out_path, args.valid_file, args.j)

    if args.command == "test":
        test(args.test_file, args.model_weights, args.out_path, config, args.j)

    if args.command == "pred":
        pred(
            args.pred_file, 
            model_weights=args.model_weights, 
            out_path=args.out_path, 
            logits=args.logits, 
            config=config, 
            nworkers=args.j, 
            draw=args.draw, 
            draw_resolution=args.draw_resolution
        )


def train_kfold(train_file, config={}, out_path=None, n_folds=5, nworkers=2, verbose=True):
    """
    Perform k-fold cross-validation training.
    
    Args:
        train_file: Path to training data CSV
        config: Model configuration dictionary
        out_path: Output directory path
        n_folds: Number of folds
        nworkers: Number of data loader workers
        verbose: Whether to print progress
    """
    if out_path is None:
        out_path = f"cv_results_{str(datetime.today()).replace(' ', '-')}/"
    
    if verbose:
        print(f"Starting {n_folds}-fold cross-validation")
        print("Working on", out_path)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    else:
        raise ValueError(f"Output path {out_path} already exists")

    data = pd.read_csv(train_file)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\n{'='*50}")
        print(f"Running Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*50}\n")
        
        fold_dir = os.path.join(out_path, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        fold_train_file = os.path.join(fold_dir, "train.csv")
        fold_val_file = os.path.join(fold_dir, "valid.csv")
        
        train_data.to_csv(fold_train_file, index=False)
        val_data.to_csv(fold_val_file, index=False)
        
        fold_results_dir = os.path.join(fold_dir, "results")
        try:
            fold_metrics = train(fold_train_file, config, fold_results_dir, fold_val_file, nworkers, verbose)
            if fold_metrics:
                fold_metrics['fold'] = fold_idx + 1
                all_metrics.append(fold_metrics)
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {e}")
            continue
        
        # Clean up temporary files
        if os.path.exists(fold_train_file):
            os.remove(fold_train_file)
        if os.path.exists(fold_val_file):
            os.remove(fold_val_file)
    
    if all_metrics:
        collect_cv_results(out_path, n_folds, verbose)
    else:
        print("No successful folds completed.")


def collect_cv_results(base_output, n_folds, verbose=True):
    """Collect results from all folds and create summary."""
    all_metrics = []
    
    for fold in range(1, n_folds + 1):
        log_file = os.path.join(base_output, f"fold_{fold}", "results", "train_log.csv")
        if os.path.exists(log_file):
            try:
                fold_metrics = pd.read_csv(log_file)
                if 'valid_f1_post' in fold_metrics.columns:
                    best_idx = fold_metrics['valid_f1_post'].idxmax()
                elif 'valid_f1' in fold_metrics.columns:
                    best_idx = fold_metrics['valid_f1'].idxmax()
                else:
                    print(f"Warning: No valid F1 metrics found in fold {fold}")
                    continue
                    
                best_metrics = fold_metrics.loc[best_idx].copy()
                best_metrics['fold'] = fold
                all_metrics.append(best_metrics)
            except Exception as e:
                print(f"Error reading fold {fold} results: {e}")
    
    if not all_metrics:
        print("No valid results found to summarize.")
        return
    
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(base_output, "cv_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    if verbose:
        print("\nCross-Validation Results:")
        print("="*50)
        
        metrics_to_report = ['valid_f1', 'valid_f1_post', 'valid_precision_post', 'valid_recall_post']
        for metric in metrics_to_report:
            if metric in summary_df.columns:
                mean_val = summary_df[metric].mean()
                std_val = summary_df[metric].std()
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nDetailed results saved to: {summary_path}")


def train(train_file, config={}, out_path=None, valid_file=None, nworkers=2, verbose=True):
    """
    Train the Q1Fold model.
    
    Args:
        train_file: Path to training data CSV
        config: Model configuration dictionary
        out_path: Output directory path
        valid_file: Path to validation data CSV (optional)
        nworkers: Number of data loader workers
        verbose: Whether to print progress
        
    Returns:
        Dictionary of best validation metrics
    """
    if out_path is None:
        out_path = f"results_{str(datetime.today()).replace(' ', '-')}/"

    if verbose:
        print("Working on", out_path)

    if "cache_path" not in config:
        config["cache_path"] = "cache/"
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    else:
        raise ValueError(f"Output path {out_path} already exists")

    if valid_file is None:
        data = pd.read_csv(train_file)
        valid_split = config.get("valid_split", 0.1)
        train_file_path = os.path.join(out_path, "train.csv")
        valid_file = os.path.join(out_path, "valid.csv")

        val_data = data.sample(frac=valid_split)
        val_data.to_csv(valid_file, index=False)
        data.drop(val_data.index).to_csv(train_file_path, index=False)
        train_file = train_file_path
        
    batch_size = config.get("batch_size", 4)
    train_loader = DataLoader(
        SeqDataset(train_file, training=True, **config),
        batch_size=batch_size, 
        shuffle=True,
        num_workers=nworkers,
        collate_fn=pad_batch
    )
    valid_loader = DataLoader(
        SeqDataset(valid_file, **config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )

    net = q1fold(train_len=len(train_loader), **config)
    
    best_f1, patience_counter = -1, 0
    patience = config.get("patience", 30)
    if verbose:
        print("Start training...")
    max_epochs = config.get("max_epochs", 1000)
    logfile = os.path.join(out_path, "train_log.csv") 
    
    best_metrics = None
        
    for epoch in range(max_epochs):
        net.current_epoch = epoch

        train_metrics = net.fit(train_loader)
        val_metrics = net.test(valid_loader)

        # Monitor quantum parameters (handles both legacy QuantumConv1D
        # and the new MultiScaleQuantumConv1D layouts gracefully)
        if epoch % 1 == 0:
            # Legacy 6-qubit module: extractor sits at resnet1d[0]
            legacy_extractor = None
            try:
                legacy_extractor = net.resnet1d[0]
            except (TypeError, IndexError):
                legacy_extractor = None

            if legacy_extractor is not None and hasattr(legacy_extractor, 'theta'):
                theta_norm = legacy_extractor.theta.norm().item()
                embed_norm = legacy_extractor.embed_weights.norm().item()

                theta_grad_norm = legacy_extractor.theta.grad.norm().item() if legacy_extractor.theta.grad is not None else 0
                embed_grad_norm = legacy_extractor.embed_weights.grad.norm().item() if legacy_extractor.embed_weights.grad is not None else 0

                if verbose:
                    print(f"\n[Epoch {epoch}] Quantum Parameter Norms:")
                    print(f"  Theta norm: {theta_norm:.4f} (grad: {theta_grad_norm:.6f})")
                    print(f"  Embed norm: {embed_norm:.4f} (grad: {embed_grad_norm:.6f})")
            else:
                # MultiScaleQuantumConv1D: per-branch theta/embed_weights/pos_alpha
                multiscale = getattr(net, 'resnet1d', None)
                branches = getattr(multiscale, 'branches', None)
                if branches is not None and verbose:
                    print(f"\n[Epoch {epoch}] Multi-scale quantum parameter norms:")
                    for k, branch in zip(multiscale.scales, branches):
                        theta = branch.theta
                        embed = branch.embed_weights
                        pos = branch.pos_alpha
                        tg = theta.grad.norm().item() if theta.grad is not None else 0.0
                        eg = embed.grad.norm().item() if embed.grad is not None else 0.0
                        pg = pos.grad.norm().item()   if pos.grad   is not None else 0.0
                        print(f"  k={k}: theta {theta.norm().item():.4f} (g {tg:.6f}) | "
                              f"embed {embed.norm().item():.4f} (g {eg:.6f}) | "
                              f"pos_a {pos.norm().item():.4f} (g {pg:.6f})")
        
        current_f1 = val_metrics.get("f1_post", val_metrics.get("f1", -1))
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            tr.save(net.state_dict(), os.path.join(out_path, "weights.pmt"))
            best_metrics = {**train_metrics, **{f"valid_{k}": v for k, v in val_metrics.items()}}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                break
        
        if not os.path.exists(logfile):
            with open(logfile, "w") as f: 
                msg = ','.join(['epoch']+[f"train_{k}" for k in sorted(train_metrics.keys())]+[f"valid_{k}" for k in sorted(val_metrics.keys())]) + "\n"
                f.write(msg)
                f.flush()
                if verbose:
                    print(msg)

        with open(logfile, "a") as f: 
            msg = ','.join([str(epoch)]+[f'{train_metrics[k]:.4f}' for k in sorted(train_metrics.keys())]+[f'{val_metrics[k]:.4f}' for k in sorted(val_metrics.keys())]) + "\n"
            f.write(msg)
            f.flush()    
            if verbose:
                print(msg)
            
    # Clean up
    shutil.rmtree(config["cache_path"], ignore_errors=True)
    
    tmp_file = os.path.join(out_path, "train.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    tmp_file = os.path.join(out_path, "valid.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    
    return best_metrics


def test(test_file, model_weights=None, output_file=None, config={}, nworkers=2, verbose=True):
    """
    Test the model on a dataset.
    
    Args:
        test_file: Path to test data CSV
        model_weights: Path to model weights
        output_file: Path to save metrics
        config: Model configuration
        nworkers: Number of data loader workers
        verbose: Whether to print progress
    """
    test_file = validate_file(test_file)
    if "verbose" not in config:
        config["verbose"] = verbose

    test_loader = DataLoader(
        SeqDataset(test_file, **config),
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )

    if model_weights is not None:
        net = q1fold(weights=model_weights, **config)
    else:
        net = q1fold(pretrained=True, **config)
    
    if verbose:
        print(f"Start test of {test_file}")        
    test_metrics = net.test(test_loader)
    summary = ",".join([k for k in sorted(test_metrics.keys())]) + "\n" + ",".join([f"{test_metrics[k]:.3f}" for k in sorted(test_metrics.keys())]) + "\n" 
    
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(summary)
    if verbose:
        print(summary)


def pred(pred_input, sequence_id='pred_id', model_weights=None, out_path=None, logits=False, 
         config={}, nworkers=2, draw=False, draw_resolution=10, verbose=True):
    """
    Generate predictions for sequences.
    
    Args:
        pred_input: Input file path or sequence string
        sequence_id: ID for sequence if provided as string
        model_weights: Path to model weights
        out_path: Output path for predictions
        logits: Whether to output logits
        config: Model configuration
        nworkers: Number of data loader workers
        draw: Whether to draw structures
        draw_resolution: Drawing resolution
        verbose: Whether to print progress
    """
    if out_path is None:
        output_format = "text"
    else:
        _, ext = os.path.splitext(out_path)
        if ext == "":
            if os.path.isdir(out_path):
                raise ValueError(f"Output path {out_path} already exists")
            os.makedirs(out_path)
            output_format = "ct"
        elif ext != ".csv":
            raise ValueError(f"Output path must be a .csv file or a folder, not {ext}")
        else:
            output_format = "csv"

    file_input = os.path.isfile(pred_input)
    if file_input:
        pred_file = validate_file(pred_input)
    else:
        pred_input = pred_input.upper().strip()
        nt_set = set([i for item in list(NT_DICT.values()) for i in item] + list(NT_DICT.keys()))
        if set(pred_input).issubset(nt_set):
            pred_file = f"{sequence_id}.csv"
            with open(pred_file, "w") as f:
                f.write("id,sequence\n")
                f.write(f"{sequence_id},{pred_input}\n")
        else:
            raise ValueError(f"Invalid input nt {set(pred_input)}, either the file is missing or the sequence has invalid nucleotides (should be any of {nt_set})")
    
    pred_loader = DataLoader(
        SeqDataset(pred_file, for_prediction=True, **config),
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch,
    )
    
    if model_weights is not None:
        net = q1fold(weights=model_weights, **config)
    else:
        net = q1fold(pretrained=True, **config)

    if verbose:        
        print(f"Start prediction of {pred_file}")

    predictions, logits_list = net.pred(pred_loader, logits=logits)
    
    if draw:
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            ctfile = "tmp.ct"
            write_ct(ctfile, item.id, item.sequence, item.base_pairs)
            dotbracket = ct2dot(ctfile)
            
            png_file = item.id + ".png"
            if out_path is not None and os.path.isdir(out_path):
                png_file = os.path.join(out_path, png_file)
            if dotbracket:
                dot2png(png_file, item.sequence, dotbracket, resolution=draw_resolution)
            ct2svg("tmp.ct", png_file.replace(".png", ".svg"))

    if not file_input:
        os.remove(pred_file)

    if output_format == "text":
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            ctfile = "tmp.ct"
            write_ct(ctfile, item.id, item.sequence, item.base_pairs)
            dotbracket = ct2dot(ctfile)
            print(item.id)
            print(item.sequence)
            print(dotbracket)
            print()
    elif output_format == "csv":
        predictions.to_csv(out_path, index=False)
    else:  # ct
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            write_ct(os.path.join(out_path, str(item.id) + ".ct"), item.id, item.sequence, item.base_pairs)
    
    if logits:
        base = os.path.split(out_path)[0] if not os.path.isdir(out_path) else out_path
        if len(base) == 0:
            base = "."
        out_path_dir = base + "/logits/"
        os.mkdir(out_path_dir)
        for id, pred_mat, pred_post in logits_list:
            pickle.dump((pred_mat, pred_post), open(os.path.join(out_path_dir, id + ".pk"), "wb"))


if __name__ == "__main__":
    main()
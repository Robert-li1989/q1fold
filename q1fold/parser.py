# Adapted from sincFold (https://github.com/sinc-lab/sincFold), GPL-3.0.
# Modifications for Q1Fold (c) 2026 Zeyu Li et al., GPL-3.0.
"""
Q1Fold Command Line Argument Parser

Provides argument parsing for training, testing, and prediction commands.
"""

import argparse
from ._version import __version__


def parser():
    """Create and return the argument parser."""
    
    main_parser = argparse.ArgumentParser(
        prog="q1fold",
        description="Q1Fold: A hybrid quantum-classical CNN for RNA secondary structure prediction",
        epilog="https://github.com/Robert-li1989/q1fold",
    )
    main_parser.add_argument("-c", "--config", type=str, help="Config file (optional, overrides any other options)")
    main_parser.add_argument("-d", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    main_parser.add_argument("-batch", type=int, default=4, help="Batch size for handling sequences")
    main_parser.add_argument("-j", type=int, default=2, help="Number of parallel workers")
    main_parser.add_argument("--quiet", action="store_true", help="Quiet mode (default: False)") 
    main_parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length to process (default: 512)") 
    main_parser.add_argument('--version', '-v', action='version', version='%(prog)s ' + __version__)

    subparsers = main_parser.add_subparsers(
        title="Actions", dest="command", description="Running commands", required=False
    )

    # Train parser
    parser_train = subparsers.add_parser("train", help="Train a new model")
    parser_train.add_argument(
        "train_file",
        type=str,
        help="Training dataset (csv file with 'id', 'sequence', and 'dotbracket' or 'base_pairs' columns)",
    )
    parser_train.add_argument(
        "--valid-file",
        type=str,
        help="Validation dataset to stop training. If not provided, validation split is randomly generated from training data.",
    )
    parser_train.add_argument(
        "-f", "--folds",
        type=int,
        default=None,
        help="Number of folds for k-fold cross-validation. If specified, --valid-file is ignored.",
    )
    parser_train.add_argument(
        "-o",
        type=str,
        dest="out_path",
        help="Output path (if not provided, it is generated with the current date)",
    )
    parser_train.add_argument(
        "-n", "--max-epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs",
    )
    main_parser.add_argument(
        "--no-cache",
        action="store_true", 
        help="Disable cache of data for training (default: cache is used)",
    )

    # Test parser
    parser_test = subparsers.add_parser("test", help="Test a model")
    parser_test.add_argument(
        "test_file",
        type=str,
        help="Testing dataset (csv file with 'id', 'sequence', and 'dotbracket' or 'base_pairs' columns)",
    )
    parser_test.add_argument(
        "-w", type=str, dest="model_weights", help="Trained model weights"
    )
    parser_test.add_argument(
        "-o",
        type=str, dest="out_path", 
        help="Output test metrics (default: only printed on the console)",
    )

    # Prediction parser
    parser_pred = subparsers.add_parser(
        "pred", help="Predict structures for a list of sequences"
    )
    parser_pred.add_argument(
        "--name", type=str, default="console_input", dest="sequence_name", 
        help="Sequence name (default: console_input)"
    )
    parser_pred.add_argument(
        "pred_file",
        type=str,
        help="Dataset to predict. It can be a csv file with 'id' and 'sequence' columns or a fasta file",
    )
    parser_pred.add_argument(
        "-o",
        type=str, dest="out_path", 
        help="Output path: .csv file or directory for CT files (default: pred.csv)",
    )
    parser_pred.add_argument(
        "--logits", action="store_true", 
        help="Output scores instead of hard predictions (default: False)"
    )
    parser_pred.add_argument(
        "-w", type=str, dest="model_weights", help="Trained model weights"
    )
    parser_pred.add_argument(
        "--draw", action="store_true", help="Draw structure (default: False)"
    ) 
    parser_pred.add_argument(
        "--draw-resolution", default=5, type=int,
        help="Drawing resolution (default: 5 times the base)"
    ) 

    return main_parser.parse_args()

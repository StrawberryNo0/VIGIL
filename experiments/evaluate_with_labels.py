#!/usr/bin/env python3
"""Evaluate detector with ground truth labels.

Expects files named as: {category}_{id}.wav
Example: bonafide_001.wav, synthetic_042.wav

Usage:
    python evaluate_with_labels.py --model wav2vec2-spoofing --input-dir ./data/labeled_audio --output-dir ./results
    python evaluate_with_labels.py --model wav2vec2-spoofing --input-dir ./data/labeled_audio --output-dir ./results --threshold 0.7
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vigil.detectors import get_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_label_from_filename(filename: str) -> str:
    """Extract ground truth label from filename.
    
    Args:
        filename: Filename like "bonafide_001.wav" or "synthetic_042.wav"
        
    Returns:
        Label string ("bonafide" or "synthetic") or None if unparseable
    """
    name = Path(filename).stem  # Remove extension
    parts = name.split("_")
    
    if parts[0] in ["bonafide", "synthetic"]:
        return parts[0]
    
    return None


def evaluate_with_labels(
    model_name: str,
    input_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    device: str = "cpu"
) -> None:
    """Evaluate detector on labeled audio files with metrics.
    
    Args:
        model_name: Name of detector ("wav2vec2-spoofing")
        input_dir: Path to directory containing labeled WAV files
        output_dir: Path to write results
        threshold: Classification threshold (default: 0.5)
        device: "cpu" or "cuda"
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Threshold: {threshold}")
    
    # Find audio files
    audio_files = sorted(input_path.rglob("*.wav"))
    if not audio_files:
        logger.warning(f"No WAV files found in {input_path}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Parse labels
    labeled_files = []
    for audio_file in audio_files:
        label = parse_label_from_filename(audio_file.name)
        if label is None:
            logger.warning(f"Could not parse label from {audio_file.name}, skipping")
            continue
        labeled_files.append((audio_file, label))
    
    if not labeled_files:
        logger.error("No files with valid labels found")
        return
    
    logger.info(f"Parsed labels for {len(labeled_files)} files")
    bonafide_count = sum(1 for _, l in labeled_files if l == 'bonafide')
    synthetic_count = sum(1 for _, l in labeled_files if l == 'synthetic')
    logger.info(f"  Bonafide: {bonafide_count}, Synthetic: {synthetic_count}")
    
    # Check class imbalance
    total = len(labeled_files)
    if bonafide_count / total < 0.1 or synthetic_count / total < 0.1:
        logger.warning(
            f"Class imbalance detected: {bonafide_count} bonafide, {synthetic_count} synthetic. "
            f"Accuracy may be misleading. Use ROC-AUC for model comparison."
        )
    
    # Initialize detector
    logger.info(f"Initializing {model_name} detector...")
    try:
        detector = get_detector(model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return
    
    # Run inference
    results = []
    errors = []
    y_true = []
    y_pred = []
    y_pred_proba_synthetic = []
    
    for i, (audio_file, ground_truth) in enumerate(labeled_files, 1):
        logger.info(f"[{i}/{len(labeled_files)}] {audio_file.name}")
        
        try:
            result = detector.detect(str(audio_file))
            result["filename"] = audio_file.name
            result["ground_truth"] = ground_truth
            result["predicted_class"] = (
                "synthetic" if result["synthetic_probability"] > threshold else "bonafide"
            )
            result["correct"] = result["predicted_class"] == ground_truth
            results.append(result)
            
            # Collect metrics
            y_true.append(1 if ground_truth == "synthetic" else 0)
            y_pred.append(1 if result["predicted_class"] == "synthetic" else 0)
            y_pred_proba_synthetic.append(result["synthetic_probability"])
            
            status = "✓" if result["correct"] else "✗"
            logger.info(
                f"  {status} Pred: {result['predicted_class']}, "
                f"Synthetic prob: {result['synthetic_probability']:.3f}"
            )
        
        except Exception as e:
            logger.error(f"  Error: {str(e)}")
            errors.append({
                "filename": audio_file.name,
                "ground_truth": ground_truth,
                "error": str(e)
            })
    
    if not results:
        logger.error("No successful detections")
        return
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba_synthetic)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Write results
    results_file = output_path / f"{model_name}_labeled_results.jsonl"
    logger.info(f"Writing detailed results to {results_file}")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Write metrics
    metrics = {
        "model": model_name,
        "device": device,
        "threshold": threshold,
        "total_samples": len(labeled_files),
        "successful": len(results),
        "errors": len(errors),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "roc_auc": float(roc_auc)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "per_class_stats": {
            "bonafide": {
                "ground_truth_count": int(np.sum(y_true == 0)),
                "predicted_count": int(np.sum(y_pred == 0))
            },
            "synthetic": {
                "ground_truth_count": int(np.sum(y_true == 1)),
                "predicted_count": int(np.sum(y_pred == 1))
            }
        }
    }
    
    metrics_file = output_path / f"{model_name}_labeled_metrics.json"
    logger.info(f"Writing metrics to {metrics_file}")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Write errors
    if errors:
        errors_file = output_path / f"{model_name}_labeled_errors.jsonl"
        logger.info(f"Writing {len(errors)} errors to {errors_file}")
        with open(errors_file, "w") as f:
            for error in errors:
                f.write(json.dumps(error) + "\n")
    
    # Print report
    logger.info("\n" + "="*70)
    logger.info(f"EVALUATION METRICS ({model_name})")
    logger.info("="*70)
    logger.info(f"Total samples: {len(labeled_files)}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Threshold: {threshold}")
    logger.info("-" * 70)
    logger.info(f"Accuracy:   {accuracy:.4f}")
    logger.info(f"Precision:  {precision:.4f}")
    logger.info(f"Recall:     {recall:.4f}")
    logger.info(f"F1-Score:   {f1:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"ROC-AUC:    {roc_auc:.4f}")
    logger.info("-" * 70)
    logger.info("Confusion Matrix:")
    logger.info(f"  TN: {tn}  FP: {fp}")
    logger.info(f"  FN: {fn}  TP: {tp}")
    logger.info("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate detector on labeled audio files"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["wav2vec2-spoofing"],
        help="Model to use for detection"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing labeled WAV files"
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to write results (default: ./results)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for synthetic (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for inference (default: cpu)"
    )
    
    args = parser.parse_args()
    
    evaluate_with_labels(
        model_name=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device=args.device
    )


if __name__ == "__main__":
    main()

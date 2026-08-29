#!/usr/bin/env python3
"""Evaluate deepfake detectors on a directory of audio files.

Usage:
    python evaluate_detector.py --model wav2vec2-spoofing --input-dir ./data/test_audio --output-dir ./results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vigil.detectors import get_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_wav_files(input_dir: Path) -> List[Path]:
    """Recursively find all WAV files in a directory.
    
    Args:
        input_dir: Root directory to search
        
    Returns:
        List of Path objects for WAV files
    """
    return sorted(input_dir.rglob("*.wav"))


def evaluate_detector(
    model_name: str,
    input_dir: str,
    output_dir: str,
    device: str = "cpu"
) -> None:
    """Run detector on all audio files in input directory.
    
    Args:
        model_name: Name of detector ("wav2vec2-spoofing" or "rawnet2")
        input_dir: Path to directory containing WAV files
        output_dir: Path to write results
        device: "cpu" or "cuda"
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    
    # Find audio files
    audio_files = find_wav_files(input_path)
    if not audio_files:
        logger.warning(f"No WAV files found in {input_path}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
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
    
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            result = detector.detect(str(audio_file))
            result["filename"] = audio_file.name
            result["filepath"] = str(audio_file)
            result["predicted_class"] = (
                "synthetic" if result["synthetic_probability"] > 0.5 else "bonafide"
            )
            results.append(result)
            
            logger.info(
                f"  → {result['predicted_class'].upper()} "
                f"(synthetic: {result['synthetic_probability']:.3f}, "
                f"latency: {result['latency_ms']:.1f}ms)"
            )
        
        except Exception as e:
            logger.error(f"  → Error: {str(e)}")
            errors.append({
                "filename": audio_file.name,
                "filepath": str(audio_file),
                "error": str(e)
            })
    
    # Write results
    results_file = output_path / f"{model_name}_results.jsonl"
    logger.info(f"Writing results to {results_file}")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Write summary
    summary_file = output_path / f"{model_name}_summary.json"
    summary = {
        "model": model_name,
        "device": device,
        "total_files": len(audio_files),
        "successful": len(results),
        "errors": len(errors),
        "synthetic_count": sum(1 for r in results if r["predicted_class"] == "synthetic"),
        "bonafide_count": sum(1 for r in results if r["predicted_class"] == "bonafide"),
        "mean_latency_ms": (
            sum(r["latency_ms"] for r in results) / len(results)
            if results else 0
        ),
        "mean_synthetic_probability": (
            sum(r["synthetic_probability"] for r in results) / len(results)
            if results else 0
        )
    }
    
    logger.info(f"Writing summary to {summary_file}")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Write errors
    if errors:
        errors_file = output_path / f"{model_name}_errors.jsonl"
        logger.info(f"Writing {len(errors)} errors to {errors_file}")
        with open(errors_file, "w") as f:
            for error in errors:
                f.write(json.dumps(error) + "\n")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info(f"EVALUATION SUMMARY ({model_name})")
    logger.info("="*60)
    logger.info(f"Total files processed: {len(audio_files)}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Synthetic detected: {summary['synthetic_count']}")
    logger.info(f"Bonafide detected: {summary['bonafide_count']}")
    logger.info(f"Mean inference latency: {summary['mean_latency_ms']:.1f}ms")
    logger.info(f"Mean synthetic probability: {summary['mean_synthetic_probability']:.3f}")
    logger.info("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection models on audio files"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["wav2vec2-spoofing", "rawnet2"],
        help="Model to use for detection"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing WAV files to process"
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to write results (default: ./results)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for inference (default: cpu)"
    )
    
    args = parser.parse_args()
    
    evaluate_detector(
        model_name=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()

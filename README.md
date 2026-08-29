# VIGIL: Voice Integrity & Impersonation Guard

A minimal ML experimentation framework for evaluating pretrained speech deepfake detection models.

## Overview

VIGIL tests whether state-of-the-art deepfake detectors can distinguish bona-fide human speech from AI-generated/synthetic speech using CPU inference on readily available pretrained models.

## Models Tested

### 1. Wav2Vec2-based Spoofing Detector
- **Source**: Hugging Face Model Hub (`aniemore/wav2vec2-xlsr-multilingual-speech-spoofing-detection`)
- **Architecture**: Fine-tuned Wav2Vec2 on multilingual spoofing detection
- **Input**: 16kHz mono WAV
- **Output**: Binary classification (bonafide/spoof)
- **Inference**: CPU-compatible

### 2. RawNet2 Spoofing Detector  
- **Source**: Hugging Face Model Hub (`ashraq/rawnet2-speaker-verification`)
- **Architecture**: RawNet2 for speaker verification (embeddings for spoofing detection)
- **Input**: 16kHz mono WAV
- **Output**: Embeddings for similarity-based classification
- **Inference**: CPU-compatible

## Installation

### Requirements
- Python 3.8+
- CPU with ~4GB RAM minimum
- ~2GB disk space for models

### Setup

```bash
# Clone repository
git clone https://github.com/StrawberryNo0/VIGIL.git
cd VIGIL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Inference

```bash
# Evaluate a single model on a directory of audio files
python experiments/evaluate_detector.py \
  --model wav2vec2-spoofing \
  --input-dir ./data/test_audio \
  --output-dir ./results

# Using RawNet2 model
python experiments/evaluate_detector.py \
  --model rawnet2 \
  --input-dir ./data/test_audio \
  --output-dir ./results
```

### Input Format

**Supported audio formats**: WAV (PCM)  
**Sample rate**: 16 kHz (will be resampled if different)  
**Duration**: Minimum 0.5 seconds, recommended 1-5 seconds  
**Channels**: Mono or stereo (will be converted to mono)

### Output

For each input file, generates:
```json
{
  "filename": "sample.wav",
  "predicted_class": "bonafide",
  "synthetic_probability": 0.05,
  "bonafide_probability": 0.95,
  "inference_latency_ms": 243.5,
  "model_name": "wav2vec2-spoofing"
}
```

## Project Structure

```
VIGIL/
├── README.md
├── requirements.txt
├── src/
│   └── vigil/
│       ├── __init__.py
│       ├── detectors/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract DeepfakeDetector interface
│       │   ├── wav2vec2_detector.py
│       │   └── rawnet2_detector.py
│       └── utils/
│           ├── __init__.py
│           └── audio.py             # Audio loading/preprocessing
├── experiments/
│   ├── evaluate_detector.py          # Main evaluation script
│   └── evaluate_with_labels.py       # Evaluation with ground truth labels
└── data/
    ├── test_audio/                   # Place test WAV files here
    └── results/                      # Evaluation output
```

## API Reference

### DeepfakeDetector Interface

All detectors implement this common interface:

```python
from vigil.detectors import get_detector

detector = get_detector("wav2vec2-spoofing")
result = detector.detect("path/to/audio.wav")

# Result format:
{
    "synthetic_probability": float,   # 0.0-1.0, probability of synthetic speech
    "bonafide_probability": float,    # 0.0-1.0, probability of genuine human speech
    "model_name": str,                # Name of the model used
    "latency_ms": float               # Inference time in milliseconds
}
```

## Known Limitations

1. **Model Mismatch Risk**: Training/test domain mismatch between pretraining corpus and real-world speech
2. **AI Generation Evolution**: Models trained on older synthesis methods (Tacotron2, WaveGlow) may not detect newer architectures (Glow-TTS, HiFi-GAN)
3. **Accent/Language Bias**: Some models show degraded performance on non-English or accented speech
4. **Audio Compression**: Compressed audio (MP3, etc.) may fool detectors; WAV is required
5. **White-box Adversarial Attacks**: No resistance to adversarial perturbations designed to bypass detection
6. **No Speaker Verification**: Detects synthesis vs. natural, not speaker identity/spoofing
7. **Latency Variance**: CPU inference varies based on system load and audio duration
8. **Label Availability**: Ground truth labels required for accuracy/precision/recall/F1/ROC-AUC calculation

## Expected Performance

Based on model card reports (to be validated by local evaluation):

- **Wav2Vec2 Spoofing Detector**: ~95% accuracy on ASVSPOOF2021 LA subset
- **RawNet2-based approaches**: ~92% accuracy on similar benchmarks

**Important**: These are vendor claims. Actual performance on VIGIL test set will be measured and reported.

## How to Run Evaluation with Ground Truth Labels

If your audio files follow the naming convention `{category}_{sample_id}.wav` (e.g., `bonafide_001.wav`, `synthetic_042.wav`):

```bash
python experiments/evaluate_with_labels.py \
  --model wav2vec2-spoofing \
  --input-dir ./data/labeled_audio \
  --output-dir ./results
```

Outputs metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (where applicable)

## Error Handling

The detector gracefully handles:
- ✓ Invalid audio files (skipped with warning)
- ✓ Missing files (skipped with error log)
- ✓ Unsupported sample rates (auto-resampled to 16kHz)
- ✓ Extremely short audio (<0.5s): warning issued, still processed
- ✓ Corrupt WAV headers: skipped with detailed error

## Development Notes

Models are downloaded automatically on first use to `~/.cache/huggingface/`.

To use a custom model path:
```python
detector = get_detector("wav2vec2-spoofing", model_path="/path/to/model")
```

## Next Steps for VIGIL Backend Integration

1. Package `vigil.detectors` as a standalone module
2. Add async inference wrapper for concurrent processing
3. Implement model quantization for faster CPU inference
4. Add confidence thresholding for risk scoring
5. Create model comparison benchmarking framework

## References

- ASVSPOOF datasets: https://www.asvspoof.org/
- Wav2Vec2 paper: https://arxiv.org/abs/2006.11477
- RawNet2: https://arxiv.org/abs/1911.08608

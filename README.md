# ASR NLP Keyword Pipeline

A small, interview-ready project that demonstrates an ASR pipeline with text cleanup and keyword extraction. The codebase is intentionally kept simple and understandable while separating runtime logic from training and GUI experiments.

## Architecture

Audio
→ ASR transcription
→ text normalization
→ keyword extraction
→ structured output

## Features

- speech transcription interface
- text normalization and cleanup
- deterministic keyword extraction
- small pipeline orchestration layer
- CLI entry point for local runs
- test coverage for core logic

## Tech Stack

- Python 3.11+
- standard library + lightweight rule-based NLP
- optional ASR runtime integration via Whisper-compatible backends
- pytest for regression tests
- ruff for linting

## Project Structure

```text
asr-nlp-keyword-pipeline/
├── src/
│   └── asr_nlp_pipeline/
│       ├── __init__.py
│       ├── extraction/
│       │   ├── __init__.py
│       │   └── keyword_extractor.py
│       ├── pipeline.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── text_normalizer.py
│       └── transcription/
│           ├── __init__.py
│           └── transcriber.py
├── scripts/
│   └── run_pipeline.py
├── tests/
│   ├── test_keyword_extractor.py
│   ├── test_pipeline.py
│   └── test_text_normalizer.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pytest ruff
python -m pip install -e .
```

## Running the Pipeline

```bash
python scripts/run_pipeline.py path/to/audio.wav
```

The CLI validates the input path, runs the pipeline, and prints structured JSON. You may also save the result to a file:

```bash
python scripts/run_pipeline.py path/to/audio.wav --output outputs/result.json
```

## Example Output

```json
{
  "transcript": "Customer wants to cancel policy because the premium is too high.",
  "normalized_text": "Customer wants to cancel policy because the premium is too high.",
  "keywords": ["cancel policy", "high premium"]
}
```

## Engineering Decisions

- The runtime package is intentionally small and explicit.
- GUI code and training experiments were not mixed into the runtime package.
- Hard-coded local paths were removed from the maintained core package.
- Training-related scripts remain separate from the executable pipeline.

## Limitations

- This project keeps the core logic deterministic and lightweight.
- Real ASR runs still require a valid local Whisper-compatible runtime and a usable audio file.
- The default CLI example is intentionally simple and does not download large language or speech models.

## Background

This repository originally combined runtime transcription logic, a business GUI, and training scripts in a single older codebase. The current refactor keeps the useful runtime pieces, isolates the legacy experiment code, and presents the project in a cleaner, interview-ready layout.

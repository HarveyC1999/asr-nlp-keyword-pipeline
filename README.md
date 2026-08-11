# ASR NLP Keyword Pipeline

A small, lightweight ASR + NLP pipeline for transcribing audio, cleaning transcript text, and extracting keywords from the result.

## Architecture

```text
Audio input
  → Whisper transcription
  → text normalization
  → keyword extraction
  → structured output JSON
```

If available, the repo also contains historical material under the legacy/experiment folders that is not part of the active runtime package.

## What is maintained now

This repository keeps the runtime code in:

- [src/asr_nlp_pipeline](src/asr_nlp_pipeline)
- [scripts/run_pipeline.py](scripts/run_pipeline.py)
- [tests](tests)

Legacy or historical code has been isolated under:

- [experiments/legacy](experiments/legacy)
- [app/QAUI.py](app/QAUI.py)

## Features

- real Whisper-backed transcription through the runtime CLI
- demo/mock mode for deterministic local testing
- text normalization and cleanup
- rule-based keyword extraction
- structured pipeline output
- Python package layout with pytest + ruff checks

## Tech stack

- Python 3.11+
- `faster-whisper`
- standard library for CLI and orchestration
- `pytest` for tests
- `ruff` for linting

## Repository tree

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   └── QAUI.py
├── docs/
│   └── architecture.png
├── experiments/
│   └── legacy/
├── scripts/
│   └── run_pipeline.py
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
├── tests/
│   ├── test_cli.py
│   ├── test_keyword_extractor.py
│   ├── test_pipeline.py
│   └── test_text_normalizer.py
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
└── architecture.png  (historical file; moved here during cleanup if needed)
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Runtime CLI

Real Whisper execution:

```bash
python scripts/run_pipeline.py path/to/audio.wav --model tiny --language zh
```

This uses the maintained `WhisperTranscriber`. It will load the Whisper model and transcribe the supplied audio file.

Demo/mock mode:

```bash
python scripts/run_pipeline.py path/to/audio.wav --demo
```

This uses a deterministic fake transcriber and is intended for local testing or demos, not for real speech recognition.

## Example output

```json
{
  "transcript": "Customer wants to cancel policy because the premium is too high.",
  "normalized_text": "Customer wants to cancel policy because the premium is too high.",
  "keywords": ["cancel policy", "high premium"]
}
```

## Tests

```bash
python -m pytest -v
python -m ruff check .
```

The test suite uses mock/demo behavior and does not download Whisper models during unit tests.

## Legacy and experiments

The project retains older training and GUI artifacts in the archive folders for historical reference only:

- [app/QAUI.py](app/QAUI.py)
- [experiments/legacy](experiments/legacy)

These are not part of the maintained runtime package or the default CLI path.

## Notes

- This project is intentionally small and deterministic.
- No performance claims or user-count metrics are included unless they are explicitly backed by project evidence.
- The default runtime remains lightweight and understandable rather than over-abstracted.

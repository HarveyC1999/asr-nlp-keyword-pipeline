# ASR + NLP Pipeline for Keyword Extraction

This project implements an end-to-end speech-to-text (ASR) and NLP pipeline using Whisper and Python to process unstructured audio data and extract meaningful keywords.

The system is designed to automate transcription, improve searchability, and reduce manual text processing efforts in real-world business scenarios.

---

## 🚀 Features

- Speech-to-text transcription using Whisper
- Keyword extraction and normalization
- Post-processing and correction pipeline
- Modular design for extensibility
- GUI-based demo interface

---

## 🧠 Problem

Processing audio data manually is time-consuming and inefficient.  
Business users often need to extract key information from conversations, recordings, or speech data.

This project aims to:
- Automate transcription
- Extract meaningful keywords
- Enable faster search and analysis

---

## 🏗️ Architecture

![Architecture](./architecture.png)

### Pipeline Overview

1. Audio input is processed using Whisper for speech-to-text transcription  
2. Raw transcript is cleaned and normalized using rule-based preprocessing  
3. NLP techniques are applied for keyword extraction and matching  
4. Results are stored as structured output for downstream analysis  

---

## 🛠️ Tech Stack

- Python3.9+  
- Whisper (OpenAI)
- Pyannote (Speaker Diarization)  
- NLP (custom keyword extraction, normalization)  
- Regex / rule-based processing  
- (Optional) GUI interface  

---

## 📂 Project Structure

GUI /

|── QAUI.py # GUI interface

STT /

├── STT_demo.py # Speech-to-text demo

├── correcting.py # Text correction & normalization

Tuning/

├── dataset.py # Dataset handling

├── model_conversion.py # Model conversion / tuning

README.md


---

## 📊 Results

- Achieved >95% keyword matching accuracy after fine-tuning and keyword normalization
- Reduced manual text processing effort significantly
- Designed for internal usage scenarios (300+ users)

---

## 🧪 Example

**Input (Audio):**
"Customer wants to cancel the policy due to high premium"

**Output:**

Keywords: ["cancel policy", "high premium"]


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python STT/STT_demo.py

(Optional GUI)
python GUI/QAUI.py
```
## Optional Dependencies

Some advanced features require additional setup:

- pyannote.audio (speaker diarization)
- cx_Oracle (database integration)

These are not required for basic pipeline execution.

## ⚙️ Future Improvements
Improve NLP model performance (currently rule-based + tuning)
Integrate vector search / semantic search
Deploy as API service
Add real-time streaming support

## 📌 Notes

This project focuses on practical application of ASR + NLP in business workflows rather than pure model research.

## 👤 Author

Chung-Han(Harvey) Chang

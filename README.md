# Voice-to-Sketch: AI-Based Criminal Face Generation & Mugshot Retrieval

## 📌 Project Overview
This project focuses on generating a **forensic-style facial sketch from voice input** and optionally retrieving the **Top-5 most similar mugshots** from a dataset.  
The system integrates **speech-to-text**, **image generation**, **face embedding**, and **similarity search** into an end-to-end pipeline with a web-based interface.

The application is designed to demonstrate how AI techniques can assist in **criminal identification workflows** using multimodal inputs.

---

## 🚀 Features
- Upload an audio description of a suspect
- Convert speech to text using OpenAI Whisper
- Generate a black-and-white forensic sketch using Stable Diffusion
- Optional mugshot similarity search (Top-5 results)
- Face detection, alignment, and embedding without using dlib
- Approximate nearest neighbor search using FAISS
- Interactive Streamlit web interface

---

## 🧠 System Architecture
1. **Audio Input** → Speech-to-Text  
2. **Transcription** → Prompt Engineering  
3. **Prompt** → Forensic Sketch Generation  
4. **Sketch** → Face Refinement & Embedding  
5. **Embedding** → Similarity Search over Mugshots  
6. **Results** → Display Top-5 Matches in UI  

---

## 🛠️ Tech Stack

### Language & Runtime
- Python 3 (virtual environment using `venv`)
- Runs on Windows (PowerShell)

### Frontend / UI
- **Streamlit (`app.py`)**
  - Audio upload
  - Progress indicators
  - Displays transcription, generated sketch, and Top-5 similar mugshots
  - Toggle to enable/disable mugshot similarity search

### Speech-to-Text
- **OpenAI Whisper (`transcribe.py`)**
- FFmpeg (`ffmpeg.exe`) for audio decoding
- Converts `description.wav` into English text

### Image Generation
- **Stable Diffusion v1.5** via Hugging Face Diffusers
- PyTorch for inference
- Generates a black-and-white forensic sketch (`criminal_sketch.png`)

### Face Refinement & Embedding
- **OpenCV**
  - Grayscale conversion
  - Histogram equalization
  - Noise reduction using bilateral filtering
- **facenet-pytorch**
  - MTCNN for face detection & alignment
  - InceptionResnetV1 (VGGFace2 pretrained) for 512-D embeddings

### Similarity Search
- **iDOC Mugshot Dataset**
- **FAISS (CPU)**
  - IndexHNSWFlat for approximate nearest neighbor search
- **scikit-image**
  - SSIM for perceptual similarity scoring
- Results re-ranked using embedding distance + SSIM

### Index Building
- `build_mugshot_index.py`
  - Builds FAISS index from mugshot images
  - Saves `mugshot_index.faiss` and `mugshot_meta.json`
  - Supports `--max-images` flag for faster testing

---

## 📂 Project Structure

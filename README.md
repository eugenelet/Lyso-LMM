# A Versatile Near-Infrared Fluorescent Probe for Fast Assessment of Lysosomal Status via a Large Multimodal Model

**Authors:** Chen, Rui; Lee, Eugene; Wang, Yuxin; Yadav, Aditya; Zhong, Minling; Pragti, \*; Sun, Yujie; Diao, Jiajie

---

## Overview

This repository provides the codebase accompanying the manuscript:

**"A versatile near-infrared fluorescent probe for fast assessment of lysosomal status via a large multimodal model."**

The platform enables upload, management, and automated multimodal analysis of fluorescence microscopy images for lysosomal status evaluation, powered by a large language-vision model (Google Gemma 3-27B).

---

## Features

* Web-based interface for image upload, directory browsing, and evaluation
* Automated analysis with a large multimodal transformer (Gemma 3-27B)
* Support for batch processing and evaluation questions
* Simple REST API endpoints for programmatic integration
* Supports `.jpg`, `.jpeg`, `.png`, `.bmp` images

---

## Directory Structure

```
.
├── main_server.py         # Main Flask web server (user interface, image management)
├── model_worker.py        # Model inference worker (serves inference API, runs separately)
├── library/               # User-uploaded image folders
├── eval/                  # Evaluation images and questions
├── templates/
│   └── index.html         # Web frontend template
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/eugenelet/Lyso-LMM.git
cd Lyso-LMM
```

### 2. Set up the Conda environment

```bash
conda create -n lysolmm python=3.10
conda activate lysolmm
# Install required packages
pip install flask requests Pillow transformers accelerate torch torchvision torchaudio tqdm safetensors
```

> **Note:** For GPU inference, ensure you have CUDA, cuDNN, and NVIDIA drivers installed.

### 3. Download Model Weights

This project uses [Google Gemma 3-27B](https://huggingface.co/google/gemma-3-27b-it). The weights will be downloaded on first run via HuggingFace’s Transformers library.
No extra manual download required unless you want to cache weights beforehand.

---

## Usage

### **Step 1: Start the Model Worker**

Start the model inference server in one terminal:

```bash
python model_worker.py
# Runs by default on port 8001
```

### **Step 2: Start the Main Server**

Start the web user interface in a separate terminal:

```bash
python main_server.py
# Runs by default on port 8000
```

### **Step 3: Access the Web App**

Open your browser to:

```
http://localhost:8000
```

You can now upload images, browse folders, and analyze lysosomal status with the multimodal model.


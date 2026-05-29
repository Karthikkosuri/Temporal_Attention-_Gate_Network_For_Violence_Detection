# TAG-Net: Temporal Attention Gate Network for Violence Detection

## 📌 Project Description

TAG-Net is an advanced deep learning framework developed for automated violence 
detection in real-time video surveillance systems. The system addresses one of 
the most critical limitations in existing violence detection methods — the equal 
treatment of all video frames regardless of their relevance to violent activity.

In real-world surveillance scenarios, violent behavior occurs only during specific 
short time intervals within a longer video stream. Conventional approaches based 
on 2D CNNs, handcrafted features, or standard recurrent models fail to distinguish 
these critical moments from irrelevant background content, leading to reduced 
detection accuracy and unnecessary computational overhead.

TAG-Net solves this fundamental problem by introducing a novel **Temporal Attention 
Gate (TAG)** module — a lightweight yet powerful mechanism that dynamically assigns 
importance weights to individual video frames. By focusing computational resources 
on action-critical frames and suppressing non-informative background segments, 
TAG-Net achieves superior violence detection performance while remaining efficient 
enough for near real-time deployment.

The system is built on a hybrid spatiotemporal architecture that combines the 
spatial and short-term temporal feature extraction capabilities of **3D Convolutional 
Neural Networks (3D CNNs)** with the long-range sequential modeling power of 
**Bidirectional Long Short-Term Memory (Bi-LSTM)** networks. Together with the 
TAG module, this architecture forms a complete end-to-end pipeline that requires 
no manual feature engineering and generalizes well across diverse real-world 
surveillance conditions including occlusion, lighting changes, crowd density 
variations, and rapid motion transitions.

TAG-Net was evaluated on the publicly available **UBI-Fights benchmark dataset** 
and demonstrated competitive accuracy, improved recall for violent events, and 
a high AUC score of 0.97 — outperforming traditional handcrafted feature methods 
and 2D CNN-based spatial attention approaches in terms of overall detection 
reliability and sensitivity.

---

## 🎯 Motivation and Problem Statement

Public safety in crowded environments such as airports, metro stations, shopping 
malls, sports venues, and streets depends heavily on effective video surveillance. 
With thousands of cameras generating continuous video streams, manual monitoring 
is neither practical nor reliable. Automated violence detection systems are 
therefore essential for timely incident response and crime prevention.

Existing deep learning methods for violence detection face two major limitations:

- **Frame-level uniformity:** Most models assign equal importance to every frame 
  in a video clip, causing the network to dilute its focus across irrelevant 
  background frames rather than concentrating on frames that contain actual 
  violent motion patterns.

- **Temporal modeling gaps:** 2D CNN-based methods capture only spatial features 
  from individual frames and lack the ability to model motion dynamics across 
  time. While 3D CNNs improve this, they still require additional sequence 
  modeling to capture long-range temporal dependencies.

TAG-Net addresses both of these limitations through its attention-guided 
spatiotemporal architecture, making it suitable for deployment in intelligent 
surveillance systems that demand both high accuracy and real-time efficiency.

---

## 🧠 How It Works

The TAG-Net pipeline consists of three tightly integrated components:

### 1. 3D CNN Backbone
The 3D Convolutional Neural Network processes input video clips of 32 consecutive 
frames, each resized to 112×112 pixels. Unlike 2D CNNs that operate on individual 
frames, the 3D CNN applies convolutional kernels across spatial (height × width) 
and temporal (time) dimensions simultaneously. This allows the network to capture 
motion cues, interaction patterns, and short-term action dynamics directly from 
raw video data without any handcrafted preprocessing.

The backbone consists of multiple Conv3D layers followed by Batch Normalization, 
ReLU activation, and MaxPooling3D layers with a kernel size of (1×2×2) to preserve 
temporal resolution while reducing spatial dimensions. The output is passed through 
a Global Average Pooling 3D layer to produce a compact sequence of feature vectors 
representing the temporal evolution of the video clip.

### 2. Temporal Attention Gate (TAG) Module
The TAG module is the core innovation of this project. It receives the feature 
sequence from the 3D CNN and computes frame-level attention weights through the 
following steps:

- **Motion Representation:** For each time step, the absolute difference between 
  consecutive feature vectors is computed to capture temporal changes and motion 
  intensity across frames.
- **Feature Fusion:** The original frame feature and its motion representation 
  are concatenated to form a combined descriptor that encodes both appearance 
  and dynamics.
- **Attention Score Computation:** A learnable linear transformation is applied 
  to the fused descriptor to compute a raw attention score for each frame.
- **Sigmoid Activation:** The attention scores are passed through a sigmoid 
  function to produce soft attention weights in the range [0, 1], allowing 
  multiple frames to simultaneously receive high importance — unlike softmax 
  which forces competitive selection.
- **Feature Refinement:** Each frame's feature vector is scaled by its 
  corresponding attention weight using a residual formulation, amplifying 
  important action frames while suppressing background frames.

This lightweight attention mechanism adds minimal computational overhead while 
significantly improving the model's ability to focus on the most relevant 
segments of a video sequence.

### 3. Bidirectional LSTM (Bi-LSTM)
The attention-refined feature sequence is passed to a Bidirectional LSTM layer 
with 2×256 hidden units. The Bi-LSTM processes the sequence in both forward and 
backward temporal directions, capturing contextual dependencies from past and 
future frames simultaneously. This bidirectional processing is particularly 
effective for understanding complex violent interactions that unfold over multiple 
frames and involve subtle motion overlaps.

The outputs from both directions are concatenated and passed through fully 
connected layers (512 → 128 neurons) with ReLU activation, followed by a final 
dense layer with sigmoid activation for binary classification (fight vs. non-fight).

---

## 🏗️ Model Architectures Compared

Four architectures were systematically built, trained, and evaluated:

| Model | Accuracy (%) | Precision (%) | Recall (%) | AUC | F1-Score |
|---|---|---|---|---|---|
| 3D CNN (Baseline) | 91 | 90 | 58 | 0.93 | 0.70 |
| 3D CNN + LSTM | 95 | 91 | 84 | 0.96 | 0.87 |
| 3D CNN + Bi-LSTM | 94 | 90 | 77 | 0.97 | 0.83 |
| **3D CNN + TAG + Bi-LSTM (Proposed)** | **94** | **88** | **86** | **0.97** | **0.87** |

The proposed TAG model achieves the highest recall (86%) and matches the best 
AUC (0.97), making it the most reliable for detecting actual violent events — 
which is the most critical metric in real-world surveillance.

---

## 📂 Dataset

**UBI-Fights Dataset**
- Total: 1,000 annotated video clips
- Fight class: 216 videos (physical fights, aggressive interactions)
- Non-Fight class: 784 videos (walking, talking, sitting, normal crowd behavior)
- Format: .mp4, resolution ~320×240, 30 fps, duration 2–10 seconds

**Preprocessing Pipeline:**
- Frames resized to 112×112 pixels
- 32 consecutive frames extracted per clip (uniform temporal sampling)
- Pixel values normalized to [0, 1]
- Dataset split: 80% training / 20% testing with class-balanced shuffling
- [Download the dataset used in this experiment from here !](https://www.kaggle.com/datasets/intissarziani/ubi-fightsall)
---

## 🚀 How to Use This Repository

### 1. Clone the Repository
To get a local copy of this project on your machine, open Git Bash or 
your terminal and run:

```bash
git clone https://github.com/Karthikkosuri/Temporal_Attention-_Gate_Network_For_Violence_Detection.git
cd Temporal_Attention-_Gate_Network_For_Violence_Detection
```

### 2. Install Dependencies
Install all required Python libraries:

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib jupyter
```

### 3. Download the Dataset
Download the **UBI-Fights Dataset** from Kaggle and organize it as follows:
- 🔗 [Download UBI-Fights Dataset](https://www.kaggle.com/datasets/intissarziani/ubi-fightsall)

### 4. Run the Notebook
Open and run the main Jupyter notebook to preprocess data, train models, 
and evaluate results:

```bash
jupyter notebook latest-tag.ipynb
```

### 5. Load a Pretrained Model
To directly use one of the saved pretrained models for inference 
without retraining:

```python
import torch

# Load the best proposed model
model = torch.load('3dcnn_tag_bilstm.pt')
model.eval()
```

---

### 📁 Project Structure

```
Temporal_Attention-_Gate_Network_For_Violence_Detection/
│
├── Architectures Diagram/               # Visual architecture diagrams of all models
│   ├── 3dcnn_architecture.png           # Architecture flow of baseline 3D CNN model
│   ├── 3dcnn_lstm_architecture.png      # Architecture flow of 3D CNN + LSTM model
│   ├── 3dcnn_bilstm_architecture.png    # Architecture flow of 3D CNN + Bi-LSTM model
│   └── tag_net_architecture.png         # Architecture flow of proposed TAG-Net model
│
├── 3dcnn.pt                             # Trained weights — Baseline 3D CNN model
├── 3dcnn_lstm.pt                        # Trained weights — 3D CNN + LSTM model
├── 3dcnn_bilstm.pt                      # Trained weights — 3D CNN + Bi-LSTM model
├── 3dcnn_tag_bilstm.pt                  # Trained weights — Proposed TAG-Net (best model)
│
├── latest-tag.ipynb                     # Main Jupyter notebook for training, evaluation,
│                                        # and inference of all four model architectures
│
└── README.md                            # Project documentation
```
## ⚙️ Tech Stack

| Component | Details |
|---|---|
| Language | Python 3.10 |
| Deep Learning Framework | PyTorch 2.0 |
| Training Platform | Kaggle (2× NVIDIA T4 GPUs) |
| Optimizer | Adam (learning rate = 1e-4) |
| Loss Function | Binary Cross-Entropy |
| Training Duration | 50 epochs, batch size = 4 |
| Input Size | 32 frames × 112 × 112 × 3 (RGB) |

---

## 🚀 Key Features

- ✅ **End-to-end deep learning** — no handcrafted features or manual preprocessing
- ✅ **Novel TAG module** — lightweight sigmoid-based temporal attention for 
  frame importance weighting
- ✅ **Bidirectional temporal modeling** — captures both past and future context 
  across video frames
- ✅ **High recall** — minimizes missed violent incidents, critical for real-world 
  security applications
- ✅ **Computationally efficient** — suitable for near real-time surveillance deployment
- ✅ **Robust performance** — handles occlusion, lighting variation, crowd density 
  changes, and rapid motion
- ✅ **Outperforms** traditional handcrafted feature methods (81.3% → 94–95%) 
  and 2D CNN spatial attention approaches on the UBI-Fights benchmark

---

## 📊 Comparison with State-of-the-Art

| Method | Dataset | Accuracy (%) |
|---|---|---|
| Gaussian Filtering + XGBoost (2023) | UBI-Fights | 81.3 |
| 2D CNN + CBAM Attention (2024) | UBI-Fights | 94.6 |
| HOG + PSO + Deep Learning (2024) | UBI-Fights | 89.2 |
| **Proposed: 3D CNN + TAG + Bi-LSTM** | **UBI-Fights** | **94–95** |

---

## 🔮 Future Work

- Integration of **multi-head self-attention** and **transformer-based temporal 
  encoders** for enhanced sequence modeling
- Model optimization for **edge device deployment** (embedded surveillance cameras)
- Incorporation of **multi-modal inputs** — optical flow, human pose estimation, 
  and audio signals for richer behavioral understanding
- Exploration of **adaptive clip sampling** and **dynamic frame selection** 
  techniques for improved temporal efficiency
- Extension to **multi-class violence detection** beyond binary classification

---

## 👨‍💻 Authors

- [@Karthikkosuri](https://github.com/Karthikkosuri)

# 🎭 Six-Emotion Text Classification — ML & Deep Learning (LSTM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A Natural Language Processing project that classifies English text into **six human emotions** — Joy, Fear, Anger, Love, Sadness, and Surprise — using both classical Machine Learning (Logistic Regression + TF-IDF) and Deep Learning (LSTM) approaches. Ships with an interactive **Streamlit web app** for real-time inference.

---

## 📋 Table of Contents

- [Demo](#-demo)
- [Emotion Classes](#-emotion-classes)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)
- [License](#-license)

---

## 🎬 Demo

Launch the Streamlit app and type any English sentence — the model instantly returns the predicted emotion and its confidence probability.

```
"I can't believe how amazing this day has been!"  →  😄 Joy   (0.94)
"Why would anyone do something so cruel?"         →  😡 Anger  (0.87)
"I really miss the people I used to know."        →  😢 Sadness (0.91)
```

---

## 🎯 Emotion Classes

| # | Emotion   | Emoji |
|---|-----------|-------|
| 1 | Joy       | 😄    |
| 2 | Fear      | 😨    |
| 3 | Anger     | 😡    |
| 4 | Love      | ❤️    |
| 5 | Sadness   | 😢    |
| 6 | Surprise  | 😲    |

---

## 📂 Project Structure

```
NLP-Six-Emotion-Classification/
│
├── app.py                          # Streamlit web application (main entry point)
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── notebooks/
│   └── Emotions Classification using ML and DL.ipynb   # Full training & evaluation notebook
│
├── models/                         # Pre-trained serialised model files
│   ├── logistic_regresion.pkl      # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectoriser
│   ├── label_encoder.pkl           # Label encoder (emotion ↔ integer mapping)
│   └── vocab_info.pkl              # Vocabulary metadata
│
└── data/
    └── train.txt                   # Raw training dataset (~1.6 MB)
```

---

## ⚙️ How It Works

```
Raw Text
   │
   ▼
Text Cleaning (regex, lowercase, stopword removal, Porter Stemming)
   │
   ▼
TF-IDF Vectorisation  ──►  Logistic Regression  ──►  Predicted Emotion + Probability
                                                        (used in Streamlit app)

(Notebook also explores LSTM with word embeddings for comparison)
```

**Pipeline steps:**
1. **Pre-processing** — strip non-alphabetic characters, lowercase, remove NLTK English stopwords, apply Porter Stemming.
2. **Vectorisation** — convert cleaned text to a sparse TF-IDF feature matrix.
3. **Classification** — Logistic Regression predicts one of the six emotion labels.
4. **LSTM (notebook only)** — a sequential deep-learning model trained on padded integer sequences for comparison with the classical approach.

---

## 🛠️ Installation

### 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/NLP-Six-Emotion-Classification.git
cd NLP-Six-Emotion-Classification
```

### 2 — Create and activate a virtual environment *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.15.0 requires Python ≤ 3.11. If you only want to run the Streamlit app (Logistic Regression), you can skip the TensorFlow install.

---

## 🚀 Usage

### Run the Streamlit web app
```bash
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

1. Type or paste any English text in the input box.
2. Click **Predict**.
3. The predicted emotion and confidence probability are displayed instantly.

### Explore the training notebook
Open `notebooks/Emotions Classification using ML and DL.ipynb` in Jupyter or VS Code to:
- Reproduce the full training pipeline from scratch.
- Compare Logistic Regression vs. LSTM performance.
- Visualise class distributions and confusion matrices.

```bash
jupyter notebook "notebooks/Emotions Classification using ML and DL.ipynb"
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| File | `data/train.txt` |
| Size | ~1.6 MB |
| Format | `text;label` (semicolon-separated) |
| Classes | 6 (joy, fear, anger, love, sadness, surprise) |
| Source | Public emotion-labelled English text corpus |

---

## 🤖 Models

| Model | Vectorisation | Library | Artifact |
|---|---|---|---|
| Logistic Regression | TF-IDF | scikit-learn 1.3.2 | `models/logistic_regresion.pkl` |
| LSTM | Integer sequences + Embedding | TensorFlow 2.15.0 | Trained in notebook |

---

## 📈 Results

> Training metrics are available inside the notebook. Key observations:
- **Logistic Regression** achieves strong baseline performance with fast inference — ideal for the real-time Streamlit app.
- **LSTM** captures sequential context and generally improves recall on minority emotion classes (e.g., Surprise).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it with attribution.

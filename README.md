# CS145 Recommender Systems Final Submission

This repository contains our full implementation for the CS145 Recommender Systems Competition. It includes all models and experiments developed across the three checkpoints (content-based, sequence-based, and graph-based recommenders), as well as our final submission.

---

## 🏆 Final Leaderboard Score

**Discounted Revenue**: `$2749.1060`  
**Final Model**: Graph Convolutional Network (GCN)

---

## 📁 Repository Structure

```
.
├── checkpoint1/          # Content-Based Recommenders
│   ├── LogisticRegression.py
│   ├── RandomForest.py
│   ├── GradientBoosting.py
│   └── KNN.py
│
├── checkpoint2/          # Sequence-Based Recommenders
│   ├── ar_implementation.py
│   ├── RNN.py
│   ├── LSTM.py
│   └── GRU.py
│
├── checkpoint3/          # Graph-Based Recommender
│   └── GCN.py
│
├── submission.py         # Final leaderboard submission (GCN-based)
├── requirements.txt
├── README.md             # This file
```

---

## ✅ Included Models

**Checkpoint 1 – Content-Based**
- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)

**Checkpoint 2 – Sequence-Based**
- Autoregressive N-gram Recommender
- RNN, LSTM, GRU sequence models

**Checkpoint 3 – Graph-Based**
- Graph Convolutional Network (GCN)

---

## 🚀 Final Model (GCN)

- Constructs a bipartite user–item graph
- Trains a 2-layer GCN with ReLU activations
- Uses dot product of embeddings × item price for scoring
- Trained using binary cross-entropy with negative sampling

---

## 🔧 How to Run

1. Clone the repo:
```bash
git clone https://github.com/your-team/cs145-final.git
cd cs145-final
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
Ensure that you have:
- Python 3.8+
- PyTorch
- PySpark
- Java 8+

3. Submit `submission.py` on the course leaderboard website to reproduce results.

---

## 👥 Team Info

**Team 12**  
Final Score: `$2749.1060`  
Members: Ian Turner, [Add other names if needed]

---

## 📌 Notes

- Random seeds are set for reproducibility
- Evaluation uses revenue-weighted scoring (probability × price)
- We manually tuned embedding size (64), learning rate (0.01), and trained for 5 epochs

---

## 🧠 CS145 Concepts Applied

- Feature engineering & supervised learning
- Graph neural networks and sequential modeling
- Ranking metrics: precision@k, NDCG, MRR, discounted revenue
- Trade-offs between generalization and memorization in recommendation

---

Let us know if you want the non-markdown version or a version ready for PDF appendix formatting.

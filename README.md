# CS145 Recommender Systems Competition

This repository contains our full solution for the CS145 Recommender Systems Competition. It includes content-based, sequence-based, and graph-based recommendation models, along with our final submission that achieved a leaderboard score of **$2749.1060** in discounted revenue.

---

## 🏆 Final Model: Graph Convolutional Network (GCN)

- **Discounted Revenue**: $2749.1060  
- Represents user–item interactions as a bipartite graph  
- Learns embeddings using a 2-layer GCN with ReLU activation  
- Trained with negative sampling and binary cross-entropy loss  
- Final relevance score = dot(user, item) × item price  

---

## 📁 Repository Structure

```
├── GCN.py                 # Final GCN model implementation
├── GRU.py                 # GRU-based sequential model
├── LSTM.py                # LSTM-based sequential model
├── RNN.py                 # RNN baseline
├── ar_implementation.py   # Autoregressive n-gram recommender
├── GradientBoosting.py    # Gradient Boosting content model
├── LogisticRegression.py  # Logistic Regression baseline
├── RandomForest.py        # Random Forest content model
├── KNN.py                 # k-Nearest Neighbors model
├── submission.py          # Final leaderboard submission (GCN)
├── README.md              # This file
```

---

## ✅ Implemented Models

**Content-Based**  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- KNN  

**Sequence-Based**  
- AR n-gram recommender  
- RNN / GRU / LSTM  

**Graph-Based**  
- GCN (Final Submission)

---

## 🔧 Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/your-team/cs145-recsys-team.git
cd cs145-recsys-team
```

2. Set up environment:
```bash
pip install -r requirements.txt
```
Make sure you have:
- Python 3.8+
- Java 8+ (for PySpark)

3. Run experiments using the provided scripts or submit `submission.py` to the leaderboard.

---

## 🧪 Reproducibility

To reproduce results:
- Run the appropriate model file (e.g., `python GCN.py`)
- Make sure data generators and simulators from the course package are included
- Final model is `MyRecommender` class inside `submission.py`

---

## 👥 Team Info

**Team 12**  
Final Score: **$2749.1060**  
Members: Joe Lin, Ian Turner, Elliot Lin, and Allison Chen

---

## 📚 Course Concepts

This project applied supervised learning, graph neural networks, sequence modeling, feature engineering, and ranking-based evaluation — aligning directly with CS145's focus on scalable recommendation and mining techniques.

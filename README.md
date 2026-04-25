# Early Prediction of Misinformation Before It Goes Viral

> Predict whether a Twitter rumour thread will be confirmed **false or unverified** using only the first N replies — before it spreads.

---

```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download PHEME dataset
Download from [Figshare](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078) and extract to:
```
data/pheme-rnr-annotation-v1.0/
```

### 3. Train models (offline, one-time)
```bash
python train.py --data data/pheme-rnr-annotation-v1.0
```
This parses the dataset, runs leave-one-event-out CV for all models and window sizes,
and saves trained XGBoost models.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Evaluate results
```bash
python evaluate.py
```

---

## Feature Groups

| Prefix | Group | Examples |
|--------|-------|---------|
| `txt_` | Text | VADER sentiment, uncertainty words, negation count |
| `tmp_` | Temporal | Reply velocity, burst score, time-to-first-reply |
| `str_` | Structural | Tree depth, branching, structural virality |
| `usr_` | User | Max followers, verified ratio, account age |

---

## Evaluation Protocol
- **Leave-one-event-out** cross-validation across 5 PHEME events
- Label strategy: `true=0`, `false+unverified=1` (binary risk classification)
- Primary model: **XGBoost** with `scale_pos_weight` for class imbalance
- Explainability: **SHAP TreeExplainer** for global + local feature attribution

---

## Deployment
```bash
# Render / Railway / Streamlit Cloud
# Set start command to:
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

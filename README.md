# Book Recommendation System
**MTech Data Analysis Project — Khushi Agrawal (AU2444006)**

A full-featured Streamlit application implementing Content-Based Filtering, Collaborative Filtering (SVD), and a Hybrid recommender on the Book-Crossing dataset.

---

## Features

- **Content-Based Filtering** — TF-IDF vectorisation of book metadata with cosine similarity
- **Collaborative Filtering** — Truncated SVD on the user-item rating matrix
- **Hybrid Model** — Linearly weighted combination of CB and CF scores (tunable alpha)
- **User Segmentation** — K-Means clustering of users into 5 behavioural segments
- **Interactive visualisations** — Cosine similarity heatmap, predicted score distributions, hybrid score breakdown, alpha sensitivity analysis

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

Place the Book-Crossing CSV files in the same directory as `app.py`, or in the standard Kaggle path:

```
BX-Books.csv
BX-Users.csv
BX-Book-Ratings.csv
```

The app includes a **demo mode** with synthetic data if no dataset is found, so it will run without the CSVs for development purposes.

### 3. Run

```bash
streamlit run app.py
```

---

## Tabs

| Tab | Contents |
|-----|----------|
| Recommendations | Search any book and get top-N recommendations |
| Content-Based Analysis | TF-IDF heatmap + per-book similarity profile |
| Collaborative Filtering | SVD predicted score distribution + per-user top books |
| Hybrid Model | CB vs CF score breakdown + alpha sensitivity plot |
| Dataset Overview | Rating distribution, top authors, year histogram, popularity-quality scatter |
| User Segments | K-Means cluster profiles + age distribution by segment |

---

## Configuration

All key parameters are centralised in the `Config` class at the top of `app.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TFIDF_MAX_FEAT` | 5000 | TF-IDF vocabulary size |
| `MIN_USER_RATINGS` | 5 | Minimum ratings per user |
| `MIN_BOOK_RATINGS` | 3 | Minimum ratings per book |
| `SVD_COMPONENTS` | 40 | Latent factors for SVD |
| `HYBRID_ALPHA` | 0.5 | Default CB/CF blend weight |
| `N_CLUSTERS` | 5 | K-Means user segments |
| `SAMPLE_USERS` | 4000 | Users cap for tractability |

---

## Dataset

Book-Crossing dataset (Ziegler et al., 2005):  
https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset

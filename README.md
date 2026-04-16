# 📚 Book Recommendation System — Streamlit App

**MTech Data Analysis Project** by Khushi Agrawal (AU2444006)

A Streamlit app featuring Content-Based (TF-IDF), Collaborative Filtering (SVD), and Hybrid recommendation models on the Book-Crossing dataset.

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add the dataset (optional)
Download the **Book-Crossing Dataset** from Kaggle:
👉 https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset

Place these 3 CSV files in the **same folder** as `app.py`:
- `BX-Books.csv`
- `BX-Users.csv`
- `BX-Book-Ratings.csv`

> **No dataset?** The app runs in **Demo Mode** automatically using synthetic data — no setup needed!

### 3. Run the app
```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 📑 App Tabs

| Tab | Description |
|-----|-------------|
| 🔍 Get Recommendations | Pick a book → get recs via CB / SVD / Hybrid |
| 📊 Content-Based Viz   | TF-IDF cosine similarity heatmap |
| 🤝 SVD Viz             | Predicted score distribution per user |
| 🔀 Hybrid Viz          | CB vs CF score breakdown bar chart |
| 📈 Dataset Overview    | Rating distribution & top authors |

---

## 📁 Files
```
book_recommender/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── BX-Books.csv        ← (add manually from Kaggle)
├── BX-Users.csv        ← (add manually from Kaggle)
└── BX-Book-Ratings.csv ← (add manually from Kaggle)
```

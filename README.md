# Disney+ Content Quality & "Up Next" Discovery Strategy

An end-to-end Product & Data Analytics project evaluating Disney+ catalog quality (`1,052` titles) by combining Rotten Tomatoes critic ratings, Machine Learning quality imputation (`Random Forest`), and a hybrid `"Up Next"` recommendation engine (`TF-IDF + Cosine Similarity`).

🔗 **Interactive Tableau Dashboard:** https://public.tableau.com:9443/views/DisneyContentQualityStrategy/DisneyContentQualityUpNextDiscoveryStrategy?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

🔗 **Live "Up Next" Recommender App (Streamlit):** 
https://disney-plus-content-strategy-6say7rbhxhiwg65vfflo8x.streamlit.app/

---

### Business Problem & Context
In OTT streaming platforms, cold-start catalog titles (newly released or niche titles lacking critic scores) often suffer from poor placement in Row 2 (`"Up Next"`) carousels and CRM retention flows. This project addresses three product questions:
1. **Catalog Strategy:** How do Disney+ Originals compare to Legacy catalog titles across genres?
2. **Confounding Factor Analysis:** Does runtime drive quality, or is the correlation skewed by short-form documentaries?
3. **Discovery Optimization:** How can we score unrated titles and rank `"Up Next"` recommendations without relying solely on watch logs?

---

### Key Analytical Insights
- Legacy catalog titles outperform same-year Originals in average quality score (`54.9%` vs. `42.0%` for non-animation; `59.4%` vs. `48.3%` for animation), highlighting the retention value of classic IP.
- Initial EDA suggested longer runtime correlated with higher critic scores. Multivariate segmentation revealed a confounding effect: short-form documentaries (`30–60 mins`) clustered at lower quality scores, whereas feature-length films (`85–130 mins`) maintained consistently high ratings.
- Exclude short-form documentaries (`< 60 mins`) from `"Up Next"` auto-play queues immediately following feature-length movies to prevent session drop-off.

---

### ⚙️ Methodology
* **Data Wrangling & Feature Engineering (`Python`, `pandas`, `numpy`):**
  * Left-joined `disney_plus_titles.csv` with `MoviesOnStreamingPlatforms.csv` (`1,052` movies) to preserve unrated catalog items.
  * Engineered `release_lag`, `content_origin` (*Legacy* vs. *Same-Year Original*), `is_animation`, `is_documentary`, and `genre_count`.
* **Predictive Quality Scoring (`scikit-learn`):**
  * Trained a **Random Forest Classifier** to predict "Fresh" probability (`fresh_prob`) for unrated titles, achieving **87.26% Accuracy**, **0.857 F1-Score**, and **0.923 ROC-AUC** (outperforming a Logistic Regression baseline).
  * Created a unified `quality_score` combining actual Rotten Tomatoes scores with ML-predicted probabilities (`score_source` tracked for transparency).
* **Hybrid `"Up Next"` Recommender (`TF-IDF` + `Cosine Similarity`):**
  * Built a weighted ranking function (`60%` metadata similarity across `director`, `cast`, `primary_genre`, and `description` + `40%` `quality_score`) to balance topical relevance with content quality.
* **BI & Interactive Visualization (`Tableau Public`):**
  * Built a 4-view interactive dashboard (*Quality Gap*, *Runtime vs. Quality*, *Genre Performance & Volume*, and *Up Next Catalog Explorer*) with cross-filtering actions.

---

### Limitations & Future Improvements
* **Proxy Metrics:** Used Rotten Tomatoes critic scores as a quality proxy due to the absence of proprietary user watch-time and completion logs.
* **Next Steps:** Incorporate collaborative filtering once user-level engagement data is available, and deploy the `"Up Next"` function as a live **Streamlit** prototype.

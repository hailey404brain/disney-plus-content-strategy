import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Disney+ "Up Next" Recommender', page_icon="🎬", layout="wide")

@st.cache_data
def load_and_build_model():
    df = pd.read_csv("disney_tableau_movies.csv")
    df["content_soup"] = df["content_soup"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["content_soup"])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cos_sim

df, cos_sim = load_and_build_model()

# --- Header ---
st.title('🎬 Disney+ "Up Next" Hybrid Recommendation Engine')
st.markdown(
    "Hybrid ranking combining **60% Metadata Similarity** (`TF-IDF` on director, cast, genre & synopsis) "
    "with **40% ML-Imputed Quality Score** (`Random Forest`, 92.3% ROC-AUC)."
)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Tuning & Filters")
sim_weight = st.sidebar.slider("Similarity Weight vs. Quality Weight", 0.0, 1.0, 0.60, 0.05)
qual_weight = round(1.0 - sim_weight, 2)
st.sidebar.caption(f"Current Formula: **{int(sim_weight*100)}% Similarity** + **{int(qual_weight*100)}% Quality**")

exclude_short_docs = st.sidebar.checkbox("Exclude short-form documentaries (< 60 mins)", value=True)
top_n = st.sidebar.slider("Number of recommendations", 5, 15, 8)

# --- Main Selector ---
selected_title = st.selectbox("🍿 Select a movie you just finished watching:", sorted(df["title"].unique()))

if selected_title:
    idx = df[df["title"] == selected_title].index[0]
    selected_row = df.loc[idx]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Primary Genre", selected_row["primary_genre"])
    col2.metric("Origin", "Legacy" if "Legacy" in str(selected_row["content_origin"]) else "Original")
    col3.metric("Runtime", f"{int(selected_row['duration_num'])} mins")
    col4.metric("Quality Score", f"{selected_row['quality_score']*100:.1f}%", selected_row["score_source"])

    st.divider()

    # Compute Hybrid Score
    sim_scores = list(enumerate(cos_sim[idx]))
    results = []
    for i, sim in sim_scores:
        if i == idx:
            continue
        row = df.iloc[i]
        if exclude_short_docs and row["is_documentary"] == 1 and row["duration_num"] < 60:
            continue
        hybrid_score = (sim_weight * sim) + (qual_weight * row["quality_score"])
        results.append({
            "Recommended Title": row["title"],
            "_sort_score": hybrid_score,
            "Hybrid Match": f"{hybrid_score*100:.1f}%",
            "Similarity": f"{sim*100:.1f}%",
            "Quality Score": f"{row['quality_score']*100:.1f}%",
            "Score Source": row["score_source"],
            "Genre": row["primary_genre"],
            "Runtime (mins)": int(row["duration_num"]),
            "Director": row["director"]
        })

    rec_df = (
        pd.DataFrame(results)
        .sort_values(by="_sort_score", ascending=False)
        .drop(columns=["_sort_score"])
        .head(top_n)
    )
    st.subheader(f'✨ Top {top_n} "Up Next" Recommendations after *{selected_title}*')
    st.dataframe(rec_df, use_container_width=True, hide_index=True)
import streamlit as st
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# Setup
st.set_page_config(page_title="Sentiment Classifier", layout="centered")
st.title("Sentiment Classifier 😃😡")
st.markdown("""
#### Please upload English comments only.
This app requires you to **train the model before prediction**. Select your feature extraction method and upload your dataset to begin.
""")

# Sidebar instructions
with st.sidebar:
    st.header("📝 Instructions")
    st.markdown("""
    1. Choose a feature extraction method:
       - **N-Gram**: TF-IDF with unigrams to trigrams.
       - **Word2Vec**: Word embeddings with average vector.
       - **Combined**: Merge N-Gram and Word2Vec.
    2. Upload a CSV file with columns **'reviews'** and **'labels'**.
    3. Wait for training to complete.
    4. Make a prediction by:
       - Typing a comment manually.
       - Uploading another CSV file with a **'reviews'** column.
    """)
    st.markdown("---")
    st.subheader("👤 About Me")
    st.markdown("""
    **Mamadou Sanogo**  
    🎓 Computer Engineering Student – Big Data & AI  
    📧 [mamadou.sanogo@uir.ac.ma](mailto:mamadou.sanogo@uir.ac.ma)  
    🐙 [GitHub](https://github.com/sanogomamadou)  
    💼 [LinkedIn](https://linkedin.com/in/mamadou-sanogo-3b22a9263)  

🚀 Passionate about technology, artificial intelligence, and data-driven innovation.  
🤝 Open to collaborations, open-source projects, and professional opportunities !
""")


# Tokenizer and stop words
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    return [t for t in tokens if t.isalpha() and t not in stop_words]

# Feature extraction method selection
method = st.radio("Choose feature extraction method:", ["N-Gram", "Word2Vec", "Combined"])

# Upload CSV
uploaded_file = st.file_uploader("Upload your labeled training CSV (must contain 'reviews' and 'labels'):", type=["csv"])

# Load and preprocess
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'reviews' not in df.columns or 'labels' not in df.columns:
        st.error("Your CSV must contain 'reviews' and 'labels' columns.")
    else:
        df['tokens'] = df['reviews'].apply(preprocess)
        texts = [' '.join(tokens) for tokens in df['tokens']]
        y = df['labels'].values

        if method == "N-Gram":
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
            X = vectorizer.fit_transform(texts)

        elif method == "Word2Vec":
            word2vec_params = {'vector_size': 300, 'window': 10, 'min_count': 5}
            w2v_model = Word2Vec(sentences=df['tokens'], **word2vec_params, workers=4)
            X = np.array([
                np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv] or [np.zeros(300)], axis=0)
                for tokens in df['tokens']
            ])

            # PCA projection plot
            X_vis = PCA(n_components=2).fit_transform(X)
            df_pca = pd.DataFrame(X_vis, columns=['PC1', 'PC2'])
            df_pca['Sentiment'] = y
            df_pca['Sentiment'] = df_pca['Sentiment'].map({0: 'Negative', 1: 'Positive'})

            fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Sentiment',
                                 title='Word2Vec Embeddings (PCA)',
                                 labels={'PC1': 'PC1', 'PC2': 'PC2'},
                                 color_discrete_map={'Negative': 'blue', 'Positive': 'red'},
                                 opacity=0.5)
            st.plotly_chart(fig_pca)

            # t-SNE word embeddings
            words = list(w2v_model.wv.index_to_key[:500])
            word_vecs = np.array([w2v_model.wv[word] for word in words])
            tsne = TSNE(n_components=2, random_state=0, perplexity=30.0, n_iter=300)
            word_vecs_2d = tsne.fit_transform(word_vecs)

            fig_tsne = px.scatter(x=word_vecs_2d[:, 0], y=word_vecs_2d[:, 1], text=words,
                                  title="Word2Vec Embeddings (t-SNE)",
                                  labels={'x': 't-SNE 1', 'y': 't-SNE 2'})
            fig_tsne.update_traces(textposition='top center')
            fig_tsne.update_layout(hovermode='closest')
            st.plotly_chart(fig_tsne)

        elif method == "Combined":
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
            X_ngram = vectorizer.fit_transform(texts)
            word2vec_params = {'vector_size': 300, 'window': 10, 'min_count': 5}
            w2v_model = Word2Vec(sentences=df['tokens'], **word2vec_params, workers=4)
            X_w2v = np.array([
                np.mean([w2v_model.wv[word] for word in tokens if word in w2v_model.wv] or [np.zeros(300)], axis=0)
                for tokens in df['tokens']
            ])
            scaler = StandardScaler()
            X_w2v_scaled = scaler.fit_transform(X_w2v)
            X = hstack([X_ngram, X_w2v_scaled])

        # Train-test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearSVC(C=0.1 if method == "N-Gram" else 100 if method == "Word2Vec" else 1, dual=False, max_iter=10000)
        model.fit(X_train, y_train)
        st.success("✅ Model trained successfully!")

        # Prediction interface
        st.markdown("### Make predictions 🔍")
        user_input = st.text_area("Enter a comment to classify:")
        csv_input = st.file_uploader("Or upload a CSV file with a 'reviews' column:", type=["csv"], key="prediction")

        def predict_and_display(texts):
            if method == "N-Gram":
                X_pred = vectorizer.transform(texts)
            elif method == "Word2Vec":
                tokens = [preprocess(t) for t in texts]
                X_pred = np.array([
                    np.mean([w2v_model.wv[word] for word in tok if word in w2v_model.wv] or [np.zeros(300)], axis=0)
                    for tok in tokens
                ])
            elif method == "Combined":
                tokens = [preprocess(t) for t in texts]
                X_ng = vectorizer.transform([' '.join(tok) for tok in tokens])
                X_w2v = np.array([
                    np.mean([w2v_model.wv[word] for word in tok if word in w2v_model.wv] or [np.zeros(300)], axis=0)
                    for tok in tokens
                ])
                X_w2v_scaled = scaler.transform(X_w2v)
                X_pred = hstack([X_ng, X_w2v_scaled])

            preds = model.predict(X_pred)
            return ["Positive 😊" if p == 1 else "Negative 😠" for p in preds], preds

        if user_input:
            result, _ = predict_and_display([user_input])
            st.write(f"Prediction: **{result[0]}**")

        elif csv_input:
            pred_df = pd.read_csv(csv_input)
            if 'reviews' not in pred_df.columns:
                st.error("CSV must contain a 'reviews' column.")
            else:
                results, raw_preds = predict_and_display(pred_df['reviews'].tolist())
                pred_df['Sentiment'] = results
                st.dataframe(pred_df[['reviews', 'Sentiment']])

                pos_percent = (np.array(raw_preds) == 1).mean() * 100
                neg_percent = 100 - pos_percent
                st.markdown(f"**Positive: {pos_percent:.2f}% | Negative: {neg_percent:.2f}%**")

                st.download_button("Download Predictions CSV", pred_df.to_csv(index=False).encode(), file_name="predictions.csv", mime="text/csv")
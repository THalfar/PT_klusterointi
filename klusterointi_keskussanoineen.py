import nltk
nltk.download('stopwords')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import re
from scipy.sparse import csr_matrix

documents = ["Koira juoksee", "Kissa kiipeää", "Lintu lentää",
             "Kala ui", "Koira haukkuu", "Kissa naukuu",
             "kuopat kasvavat paljon", "kuopat ovat kurjia", "kuopat korjataan",
             "auto ajaa ojaan", "ojassa auto", "koodaus on ihan hauskaa"]

stop_words = set(stopwords.words('finnish'))

def preprocess(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'[^\w\s]', '', text)
    return text

documents = [preprocess(doc) for doc in documents]

# Laske TF-IDF-arvot
vectorizer = TfidfVectorizer()
X_reduced = vectorizer.fit_transform(documents)

# Suorita PCA-analyysi (valinnainen)
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X_reduced.toarray())

# Suorita K-means -klusterointi
n_clusters = 5  # valitse klusterien lukumäärä
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_reduced)

# Määritä, mihin klusteriin kukin datakohta kuuluu
labels = kmeans.labels_

# Tulosta klusterilabelit jokaista tekstiä kohden
for i, label in enumerate(labels):
    print(f'Teksti: "{documents[i]}" kuuluu klusteriin {label}')

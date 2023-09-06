import nltk
nltk.download('stopwords')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import re

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

# TF-IDF vektorisointi
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# PCA-dimensioiden vähennys 2D-visualisointia varten
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X.toarray())

# Lisätään tekstilabelit pienellä siirtymällä
offset = 0.2  
noppa = 0.2
for i, txt in enumerate(documents):
    if (np.random.rand() > noppa):
        plt.annotate(txt, (reduced_features[i, 0] + (np.random.rand(1)-1)*offset, reduced_features[i, 1] + (np.random.rand(1)-1)*offset))

x_min, x_max = np.min(reduced_features[:, 0]) - 1, np.max(reduced_features[:, 0]) + 1
y_min, y_max = np.min(reduced_features[:, 1]) - 1, np.max(reduced_features[:, 1]) + 1
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.show()

import jieba
import numpy as np
from _DPC import DPC

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, Birch

from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from stylecloud import gen_stylecloud

sourceFile = "./bing_crawl_bufa2.txt"
data = np.loadtxt(sourceFile, encoding='UTF-8', dtype=str, delimiter="%%")

StrList = []
splitData = []
titles = []
urls = []
for result in data:
    resultStr = (result[0]+result[1]).replace(" ", "")
    if len(resultStr) > 0 and result[2] not in urls:
        titles.append(result[0])
        urls.append(result[2])
        StrList.append(resultStr)

stopWords = open("stopWords.txt", encoding='UTF-8').read()
for resultStr in StrList:
    words = []
    cut_words = jieba.cut(resultStr)
    for item in cut_words:
        if item != "\r\n" and item not in stopWords:
            words.append(item)
    splitData.append(words)

# for wordList in splitData:
#     print(wordList)

vectors = []
corpus = []
for wordList in splitData:
    corpus.append(" ".join(wordList))
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

pca = PCA(n_components=2)
X = pca.fit_transform(tfidf.A)

n_clusters = 5

km = KMeans(n_clusters=5)
labels = km.fit_predict(X)

# db = DBSCAN(eps=0.05, min_samples=5, metric="cosine")
# labels = db.fit_predict(X)

# birch = Birch(n_clusters=n_clusters)
# labels = birch.fit_predict(tfidf.A)

# cosine_sim = cosine_distances(X)
#
# dpc = DPC(density="gauss", d_c_ratio=0.1, distance="precomputed")
# labels = dpc.fit_predict(X=cosine_sim)

score = silhouette_score(tfidf.A, labels)

clusters = [[] for i in range(n_clusters)]
words_in_clusters = [[] for i in range(n_clusters)]
for i, l in enumerate(labels):
    for word in splitData[i]:
        words_in_clusters[l%n_clusters].append(word)
    clusters[l%n_clusters].append(titles[i]+"->"+urls[i])

i = 1
for cluster in clusters:
    print(f"-======================cluster {i}========================")
    for website in cluster:
        print(website)
    i += 1

print("=====================silhouette_score======================")
print(score)

for index, words in enumerate(words_in_clusters):
    gen_stylecloud(text=' '.join(words),
                   # icon_name='fas fa-apple-whole',
                   font_path='msyh.ttc',
                   background_color='white',
                   output_name='kmeans_cloud'+str(index)+'.jpg',
                   )

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()

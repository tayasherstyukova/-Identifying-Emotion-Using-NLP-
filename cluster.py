# Following tutorial here:
# https://medium.com/@nisha.imagines/nlp-with-python-text-clustering-based-on-content-similarity-cae4ecffba3c

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans

# Convert sentences into vectors using TFDIF
vec = TfidfVectorizer(stop_words="english",ngram_range=(1,3))
vec.fit(train.sentence)
features=vec.transform(train.sentence)

# Apply K-means clustering on the feature vectors
clust = KMeans(init='k-means++',n_clusters=5,n_init=10)
clust.fit(features)
yhat = clust.predict(features)
train['Cluster Labels']=clust.labels_

train0 = train.loc[train['Cluster Labels'] == 10]
train1 = train.loc[train['Cluster Labels'] == 1]
train2 = train.loc[train['Cluster Labels'] == 2]
train3 = train.loc[train['Cluster Labels'] == 3]
train4 = train.loc[train['Cluster Labels'] == 4]

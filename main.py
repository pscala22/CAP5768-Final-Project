#Paul Scala
#CAP5768
#Final Project
#December 7th 2022
#Ran using Python 3.9.12

from sklearn.datasets import load_breast_cancer,load_digits,load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np

#Part 1
data = load_breast_cancer().data
pca = PCA(2)
df = pca.fit_transform(data)
kmeans = KMeans(n_clusters=10)
label = kmeans.fit_predict(df)
colors = ['violet','indigo','blue','green','yellow','orange','red','black','cyan','silver']

for i in range(10):
    plt.scatter(df[label==i,0],df[label==i,1],label=i,color=colors[i])
plt.title("Part 1")
plt.legend()
plt.show()

#Part 2
data = load_digits().data
pca = PCA(2)
df = pca.fit_transform(data)
dbscan = DBSCAN(min_samples=10,eps=1.5).fit(df)

labels = dbscan.labels_
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, 23)]

for i in range(23):
    class_member_mask = labels==i
    xy = df[class_member_mask & core_samples_mask]
    plt.scatter(xy[:,0],xy[:,1],label=i,color=colors[i])
    xy = df[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:,0],xy[:,1],color=colors[i])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=5)
plt.title("Part 2")
plt.show()

#Part 3
data = load_iris().data
pca = PCA(2)
df = pca.fit_transform(data)
ac = AgglomerativeClustering(n_clusters=5)
label = ac.fit_predict(df)
colors = ['blue','green','yellow','red','black']

for i in range(5):
    plt.scatter(df[label==i,0],df[label==i,1],label=i,color=colors[i])
plt.title("Part 3")
plt.legend()
plt.show()

#Part 4
data = load_digits().data
pca = PCA(2)
df = pca.fit_transform(data)
#kmeans
kmeans = KMeans(n_clusters=20)
label = kmeans.fit_predict(df)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, 20)]

for i in range(20):
    plt.scatter(df[label==i,0],df[label==i,1],label=i,color=colors[i])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=5)
plt.title("Part 4 Kmeans")
plt.show()

#agglomerative
ac = AgglomerativeClustering(n_clusters=20)
label = ac.fit_predict(df)
for i in range(20):
    plt.scatter(df[label==i,0],df[label==i,1],label=i,color=colors[i])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=5)
plt.title("Part 4 Agglomerative")
plt.show()

import numpy as np
import pandas as pd
import os
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle
import sys

surf_csv_file = 'select.surf.csv'
output_file = '../kmeans.3000.model'
cluster_num = 3000

df = pd.read_csv(surf_csv_file, header=None, delimiter=',')

kmeans_model = MiniBatchKMeans(n_clusters=cluster_num, verbose=1).fit(df)
print("K-means trained successfully!")

pickle.dump(kmeans_model, open(output_file, 'wb'))
print("Saved successfully! Check!")

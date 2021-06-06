import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

x,y = fetch_openml('mnist_784',version = 1,return_X_y=True)

print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

sample_per_class = 5
figure = plt.figure(figsize=(nclasses*2,(1+sample_per_class*2)))

idx_cls = 0
for cls in classes:
    idxs = np.flatnonzero(y== cls)
    idxs = np.random.choice(idxs,sample_per_class , replace = False)
    i = 0
    for idx in idxs:
        plt_idx = i * nclasses + idx_cls +1
        p = plt.subplot(sample_per_class, nclass ,plt_idx)
        p = sns.heatmap(np.reshape(X[idx],(22,30)), cmap=plt.cm.grey,
                xticklabels=False,yticklabels =False,cbar = False)
        p = plt.axis('off')
        i += 1
    idx_cls += 1

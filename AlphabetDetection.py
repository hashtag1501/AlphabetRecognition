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

roi =grey[upper_left[1]:bottom_right[1],
upper_left[0]:bottom_right[0]]

image_bw = im_pil.cconvert('L')
image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
pixel_filter = 20
min_pixel = np.percentile(image_bw_resized_invertes, pixel_filter)
image_bw_resized_inverted_scaledv= np.clip(image_bw_resized_inverted-min_pixel, 0 ,255)
max_pixel = np.max(image_bw_resized_inverted)
image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
image_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
test_pred = clf.predict(test_sample)
print("Predicted class is:",test_pred)
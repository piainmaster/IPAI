import os
from itertools import count

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#-----------------------------------------------------------------------
#VARIANCES: DATA IMPORT AND PREPARATION
#-----------------------------------------------------------------------

# Define the path to the dataset
dataset_path = "dataset"

# Load, convert to greyscale and resize images
X, y = [], []
for datasource in ["PLUS"]:         #, "PROTECT", "SCUT", "VERA"
    datasource_folder = dataset_path + "/" + datasource
    for category in ["genuine", "spoofed"]:
        category_folder = datasource_folder + "/" + category
        for filename in os.listdir(category_folder):
            img = cv2.imread(category_folder + "/" + filename)
            #img = cv2.resize(img, (100, 100))  # Resize the image to a fixed size
            X.append(img)       #todo: if no further operations on image, imread can be conducted here
            y.append(category)


# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   #todo: am Ende random_state diskutieren


#-----------------------------------------------------------------------
#KNN PARAMS
#-----------------------------------------------------------------------
"""
KNeighborsClassifier(
n_neighbors=5, -> number of neighbors, must be an uneven number *,
weights='uniform', -> all neighbors have the same weigth
algorithm='auto', -> Algorithm used to compute the nearest neighbors; todo: try out different one's at the very end
leaf_size=30, -> only necessary if algorithm is BallTree or KDTree
p=2, -> 1 = manhattan_distance (l1), 2 = euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.; todo: try out different one's at the very end
metric='minkowski', -> used metric
metric_params=None, -> dependend on used metric 
n_jobs=None)[source]¶
"""

#-----------------------------------------------------------------------
#VARIANCES: GLOBAL VARIANCE
#-----------------------------------------------------------------------

X_train_global_variance = []

# Loop over training data and calculate variances
for image in X_train:
    variance = [np.var(image)]
    X_train_global_variance.append(variance)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_global_variance, y_train)

X_test_global_variance = []
# Loop over test data and calculate variances
for image in X_test:
    variance = [np.var(image)]
    X_test_global_variance.append(variance)

# Initialize lists to store predicted categories
predicted_categories = neigh.predict(X_test_global_variance)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_categories)
print("Accuracy: {:.2f}".format(accuracy))


#-----------------------------------------------------------------------
#VARIANCES: LOCAL VARIANCE
#-----------------------------------------------------------------------


import os
import tarfile
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import entropy
from PIL import Image


# Step 1: Extract the dataset from the .tar file
def extract_dataset(tar_filename, extract_path):
    with tarfile.open(tar_filename, 'r') as tar:
        tar.extractall(extract_path)


# Step 2: Load and split the dataset into training and testing sets
def load_and_split_dataset(data_dir, test_size=0.2):
    file_list = os.listdir(data_dir)
    labels = [1 if 'real' in f else 0 for f in file_list]

    X = []
    for file in file_list:
        img = cv2.imread(os.path.join(data_dir, file))
        X.append(img)

    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


# Step 3: Feature extraction and classification methods
def classify_images(X, method):
    if method == 'histograms':
        X_features = [cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256]).flatten() for img in X]
    elif method == 'variance':
        X_features = [np.var(img) for img in X]
    elif method == 'entropy':
        X_features = [entropy(np.histogram(img, bins=256)[0]) for img in X]
    else:
        raise ValueError("Invalid method")

    return X_features


# Step 4: Evaluate the method
def evaluate_method(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    #todo: add other measures for evaluation
    return accuracy


# Step 5: Combine methods
#todo: not for now, do this later when everything works already
def combine_methods(X, methods):
    X_combined = []
    for img in X:
        features = []
        if 'histograms' in methods:
            hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256]).flatten()
            features.extend(hist)
        if 'variance' in methods:
            variance = np.var(img)
            features.append(variance)
        if 'entropy' in methods:
            entropy_value = entropy(np.histogram(img, bins=256)[0])
            features.append(entropy_value)
        X_combined.append(features)

    return X_combined


# Step 6: Main script
if __name__ == '__main__':
    tar_filename = 'dataset.tar'  #Todo: Replace with the actual dataset filename
    data_dir = 'dataset'  # Directory where the dataset is extracted
    extract_dataset(tar_filename, data_dir)

    X_train, X_test, y_train, y_test = load_and_split_dataset(data_dir)

    methods = ['histograms', 'variance', 'entropy']
    for method in methods:
        X_train_features = classify_images(X_train, method)
        X_test_features = classify_images(X_test, method)

        # Train a classifier (e.g., Random Forest) and predict
        # Replace this with your choice of classifier
        # classifier.fit(X_train_features, y_train)
        # y_pred = classifier.predict(X_test_features)

        accuracy = evaluate_method(y_test, y_pred)
        print(f"{method} accuracy: {accuracy:.2f}")

    # Combine methods
    #combined_methods = ['histograms', 'variance', 'entropy']
    #X_train_combined = combine_methods(X_train, combined_methods)
    #X_test_combined = combine_methods(X_test, combined_methods)

    # Train a classifier and predict
    # classifier.fit(X_train_combined, y_train)
    # y_pred_combined = classifier.predict(X_test_combined)

    accuracy_combined = evaluate_method(y_test, y_pred_combined)
    print(f"Combined methods accuracy: {accuracy_combined:.2f}")

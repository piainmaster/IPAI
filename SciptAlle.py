import math
import random

import numpy as np
import cv2
from pathlib import Path

from scipy.stats import wasserstein_distance
from skimage.measure import shannon_entropy
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------------------------------------------------
# DATA IMPORT
# -----------------------------------------------------------------------
def loadPLUS():
    # Define the path to the dataset
    datasource_path = "dataset/PLUS"

    data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004 = [], [], [], []

    # load dataset PLUS
    p = Path(datasource_path + "/" + "genuine")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        hist = histBin(img, k)
        data_PLUS_genuine.append(["genuine", img, hist])

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        histBin(img, k)
        data_PLUS_spoofed.append(["spoofed", img, hist])

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["003"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, k)
                    data_PLUS_003.append(["synthethic", img, hist])
        for variant in ["004"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, k)
                    data_PLUS_004.append(["synthethic", img, hist])

    return data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004



def loadSCUT():
    # Define the path to the dataset
    datasource_path = "dataset/SCUT"

    data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008 = [], [], [], []

    # load dataset PLUS
    p = Path(datasource_path + "/" + "genuine")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        hist = histBin(img, k)
        data_SCUT_genuine.append(["genuine", img, hist])

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        hist = hist = histBin(img, k)
        data_SCUT_spoofed.append(["spoofed", img, hist])

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",      # ignore "spoofed_synthethic_drit",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["007"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, k)
                    data_SCUT_007.append(["synthethic", img, hist])
        for variant in ["008"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, k)
                    data_SCUT_008.append(["synthethic", img, hist])

    return data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008



# -----------------------------------------------------------------------
# KNN
# -----------------------------------------------------------------------
def knncalc(k, list):
    knn_list = list()
    list.sort(key=lambda tup: tup[1]) #nach den Werten in der zweiten Spalte sortieren
    for i in range(k):
        knn_list.append(list[i][0]) # es Werten "spoofed" und "genuine" eingetragen
    fdist = dict(zip(*np.unique(knn_list, return_counts=True)))
    pred = list(fdist)[-1]
    return pred

def most_common_value(list):
    mcv = dict(zip(*np.unique(list, return_counts=True)))
    pred = list(mcv)[-1]
    return pred


# -----------------------------------------------------------------------
# HISTOGRAM
# -----------------------------------------------------------------------
k = 4
def histBin(img, k):
    if k<257:
        hist, bin_edges1 = np.histogram(img, bins=np.arange(0, k), density=True)
        return hist
    else:
        return "Falscher k-Wert"


def intersection_test_(hist1, hist2):
    intersection = np.minimum(hist1, hist2)
    int_area = intersection.sum() # /2 wenn die bins so aussehen bins=np.arange(0, 256, 0.5)
    return int_area

def euclidean_distance_test(hist1, hist2):
    sum = 0
    for i in range (0,256):
        sum = sum + (hist1[i][0]-hist2[i][0])**2
    eu_dist = math.sqrt(sum)
    return eu_dist

def manhattan_distance_funktion(x, y):
    return (abs(x-y))

def sum_manhattan_distance_test(hist1, hist2):
    sum = 0
    for i in range (0,256):
        sum = sum + manhattan_distance_funktion(hist1[i][0], hist2[i][0])
    ma_dist = sum
    return ma_dist

# Bild statt Histogramm wird übergeben
def earth_movers_distance_test(img1, img2):
    images = [img.ravel() for img in [img1, img2]]
    em_distance = wasserstein_distance(images[0], images[1])
    return em_distance

"""
# mit selbst programmierten knn code
conclusion_list1 # bereits sortiert für methode 1
conclusion_list2 # bereits sortiert für methode 2
knn_list1 = list()
knn_list2 = list()
k # die k-Foulds
for i in range(k):
    knn_list1.append(conclusion_list1[i][0]) #die ersten k werte anfügen
    knn_list2.append(conclusion_list2[i][0])
fdist1 = dict(zip(*np.unique(knn_list1, return_counts=True)))
fdist2 = dict(zip(*np.unique(knn_list2, return_counts=True)))
pred = list() # predictions list
pred.append(list(fdist1)[-1])
pred.append(list(fdist2)[-1])
final = dict(zip(*np.unique(pred, return_counts=True)))
prediction = list(final)[-1]
"""

# -----------------------------------------------------------------------
# ENTROPY
# -----------------------------------------------------------------------

# calculate Entropy with frames--> takes nr of frames and image path and gives back an array of entropies:
def calculate_entropies(image, num_frames):

    image = image[1]
    # Divide the image into frames
    entropies = []
    height, width = image.shape

    frame_size = width // num_frames

    for i in range(num_frames):
        start_x = i * frame_size
        end_x = (i + 1) * frame_size
        frame = image[:, start_x:end_x]
        entropy = shannon_entropy(frame)
        entropies.append(entropy)

    return entropies






# -----------------------------------------------------------------------
# VARIANCE
# -----------------------------------------------------------------------
def calculate_variances(image, num_frames):

    image = image[1]
    # Divide the image into frames
    variances = []
    height, width = image.shape

    frame_size = width // num_frames

    for i in range(num_frames):
        start_x = i * frame_size
        end_x = (i + 1) * frame_size
        frame = image[:, start_x:end_x]
        variance = np.var(frame)
        variances.append(variance)

    return variances

# Function to calculate patch variances of an image with size patch_x x patch_y
def calculate_patch_variances(image, patch_x, patch_y):
    patch_size = (patch_x, patch_y)
    variances = []
    h, w = image.shape
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            variances.append(np.var(patch))
    return variances

# -----------------------------------------------------------------------
# METHODEN-AUFRUFE
# -----------------------------------------------------------------------

data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004 = loadPLUS()
#data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008 = loadSCUT()





# -----------------------------------------------------------------------
# LEAVE ONE OUT CROSS VALIDATION
# -----------------------------------------------------------------------
def combine_list_with_genuine(list):
    current_data = zip(data_PLUS_genuine, list)
    return random.shuffle(current_data)

# combine lists data_PLUS_genuine and data_PLUS_003
current_data = combine_list_with_genuine(data_PLUS_003)

# convert data to numpy array
labels, images, histograms = zip(*current_data)
#features = [cv2.resize(img, (100, 100)) for img in images]
X = np.array(images)
y = np.array(labels)



loo = LeaveOneOut()

correct_predictions = 0

for train_index, test_index in loo.split(images, histograms, labels):

    # -----------------------------------------------------------------------
    # VARIANCE
    # -----------------------------------------------------------------------
    # calculate feature global variance

    # knn global variance

    # calculate feature patch variance

    # knn patch variance


    # -----------------------------------------------------------------------
    # ENTROPY
    # -----------------------------------------------------------------------
    # calculate feature global entropy

    # Calculate entropies for all data
    num_frames = 1
    entropy_list = []
    for img in images, labels:
        entropies = calculate_entropies(img, num_frames)
        entropy_list.append([labels, entropies])

    train, test = entropy_list[train_index], entropy_list[test_index]

    # knn global entropy


    # calculate feature patch entropy

    # knn patch entropy


    # -----------------------------------------------------------------------
    # HISTOGRAM
    # -----------------------------------------------------------------------
    # calculate feature histogram

    # knn histogram





    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # KNN classifier
    k = 5  # You can adjust the number of neighbors as needed
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)

    # Predict label for the left-out instance
    test_label = knn_classifier.predict(X_test)

    if test_label == y_test[0]:
        correct_predictions += 1

accuracy = correct_predictions / len(images)
print(f"Accuracy with Leave-One-Out Cross-Validation: {accuracy*100:.2f}%")

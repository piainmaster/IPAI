import math

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

def earth_movers_distance_test(img1, img2):
    images = [img.ravel() for img in [img1, img2]]
    em_distance = wasserstein_distance(images[0], images[1])
    return em_distance


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



# -----------------------------------------------------------------------
# METHODEN-AUFRUFE
# -----------------------------------------------------------------------

data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004 = loadPLUS()
#data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008 = loadSCUT()


# Calculate entropies for data_PLUS_genuine
num_frames = 6
entropy_list = []
for img in data_PLUS_genuine:
    entropies = calculate_entropies(img, num_frames)
    entropy_list.append(["genuine", entropies])

print("X")





# -----------------------------------------------------------------------
# LEAVE ONE OUT CROSS VALIDATION
# -----------------------------------------------------------------------


loo = LeaveOneOut()

correct_predictions = 0

for train_index, test_index in loo.split(images, labels):
    X_train, X_test = images[train_index], images[test_index]
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
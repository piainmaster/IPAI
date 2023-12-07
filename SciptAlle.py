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
        hist = histBin(img, bin)
        data_PLUS_genuine.append(["genuine", img, hist])

    """p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        histBin(img, bin)
        data_PLUS_spoofed.append(["spoofed", img, hist])"""

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["003"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, bin)
                    data_PLUS_003.append(["synthethic", img, hist])
        """for variant in ["004"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, bin)
                    data_PLUS_004.append(["synthethic", img, hist])"""

    return data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004



def loadSCUT():
    # Define the path to the dataset
    datasource_path = "dataset/SCUT"

    data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008 = [], [], [], []

    # load dataset PLUS
    p = Path(datasource_path + "/" + "genuine")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        hist = histBin(img, bin)
        data_SCUT_genuine.append(["genuine", img, hist])

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        hist = hist = histBin(img, bin)
        data_SCUT_spoofed.append(["spoofed", img, hist])

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",      # ignore "spoofed_synthethic_drit",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["007"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, bin)
                    data_SCUT_007.append(["synthethic", img, hist])
        for variant in ["008"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, bin)
                    data_SCUT_008.append(["synthethic", img, hist])

    return data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008



# -----------------------------------------------------------------------
# KNN
# -----------------------------------------------------------------------
def knncalc(k, feature_list):
    knn_list = list()
    feature_list.sort(key=lambda tup: tup[1]) #nach den Werten in der zweiten Spalte sortieren (kng)
    for i in range(k):
        knn_list.append(feature_list[i][0]) # es werden "spoofed" und "genuine" eingetragen
    fdist = dict(zip(*np.unique(knn_list, return_counts=True)))
    pred = max(fdist, key=fdist.get)
    return pred


####### for last step if we have several feature classifications
def most_common_value(list):
    mcv = dict(zip(*np.unique(list, return_counts=True)))
    pred = list(mcv)[-1]
    return pred


# -----------------------------------------------------------------------
# HISTOGRAM
# -----------------------------------------------------------------------

def histBin(img, bins):
    if bins<257:
        hist, bin_edges1 = np.histogram(img, bins=np.arange(0, bins), density=True)
        return hist
    else:
        return "Falscher k-Wert"

def intersection_test(hist1, hist2):
    intersection = np.minimum(hist1, hist2)
    int_area = intersection.sum() # /2 wenn die bins so aussehen bins=np.arange(0, 256, 0.5)
    resultat = 1-int_area
    return resultat

def euclidean_distance_test(hist1, hist2):
    sum = 0
    for i in range (0,bin):
        sum = sum + (hist1[i][0]-hist2[i][0])**2
    eu_dist = math.sqrt(sum)
    return eu_dist

def manhattan_distance_test(hist1, hist2):
    sum = 0
    for i in range (0,bin):
        sum = sum + abs(hist1[i][0] - hist2[i][0])
    ma_dist = sum
    return ma_dist

#earthmovers distance
def em_dist(img1, img2):
    images = [img.ravel() for img in [img1, img2]]
    em_distance = wasserstein_distance(images[0], images[1])
    return em_distance

# -----------------------------------------------------------------------
# ENTROPY
# -----------------------------------------------------------------------

# calculate Entropy with frames--> takes nr of frames and image path and gives back an array of entropies:
def calculate_entropies(image, num_frames):

    #image = image[1]
    # Divide the image into frames
    entropies = []
    size = image.shape
    height, width = size[0], size[1]

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
bin = 6
data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004 = loadPLUS()
#data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008 = loadSCUT()




# -----------------------------------------------------------------------
# LEAVE ONE OUT CROSS VALIDATION
# -----------------------------------------------------------------------

def combine_list_with_genuine(list):
    current_data = data_PLUS_genuine + list
    return current_data

# combine lists data_PLUS_genuine and data_PLUS_003
current_data = []
current_data = combine_list_with_genuine(data_PLUS_003)


# convert data to numpy array
labels_list, images_list, histograms_list = [], [], []
counter = 0
for row in current_data:
    labels_list.append(row[0])
    images_list.append(row[1])
    histograms_list.append(row[2])
features = [cv2.resize(img, (736,192)) for img in images_list]      #Alternative sklearn.preprocessing.StandardScaler
labels = np.array(labels_list)
images = np.array(features)
histograms = np.array(histograms_list)


# -----------------------------------------------------------------------
# ENTROPY
# -----------------------------------------------------------------------
# calculate feature global entropy
num_frames = 1
entropy_list = []
for img in range(len(images)):
    entropies = calculate_entropies(images[img], num_frames)
    entropy_list.append([labels[img], entropies])



# knn global entropy


# calculate feature patch entropy

# knn patch entropy



loo = LeaveOneOut()
pred_list = []
correct_entropy_preds = 0
correct_em_preds = 0
correct_it_preds = 0
correct_ed_preds = 0
correct_smd_preds = 0

for i, (train_index, test_index) in enumerate(loo.split(images)):
    #calculate distances
    test_entropy = entropy_list[test_index[0]]
    entropy_distances = []
    for j in train_index:
        entropy_distances.append([entropy_list[j][0], abs(entropy_list[j][1][0] - test_entropy[1][0])])

    #prediction for current test image
    k_knn = 3
    pred_entropy = knncalc(k_knn, entropy_distances)

    if pred_entropy == test_entropy[0]:
        correct_entropy_preds += 1

    # -----------------------------------------------------------------------
    # VARIANCE
    # -----------------------------------------------------------------------
    # calculate feature global variance

    # knn global variance

    # calculate feature patch variance

    # knn patch variance

    # -----------------------------------------------------------------------
    # HISTOGRAM
    # -----------------------------------------------------------------------
    # calculate feature histogram
    # calculate feature global entropy

    #earthmovers distance
    em_distance = []
    for j in range(len(images)):
        em = em_dist(images[test_index], images[j])
        em_distance.append([labels[j], em])

    # intersection distance (runs fast)
    it_distance = []
    for j in range(len(images)):
        it = intersection_test(histograms[test_index], histograms[j])
        it_distance.append([labels[j], it])

    # euclidian distance
    ed_distance = []
    for j in range(len(images)):
        ed = euclidean_distance_test(histograms[test_index], histograms[j])
        ed_distance.append([labels[j], ed])

    # sum of manhattan distances
    smd_distance = []
    for j in range(len(images)):
        smd = manhattan_distance_test(histograms[test_index], histograms[j])
        smd_distance.append([labels[j], smd])

    # prediction for current test image
    k_knn = 3
    pred_em = knncalc(k_knn, em_distance)
    pred_it = knncalc(k_knn, it_distance)
    pred_ed = knncalc(k_knn, ed_distance)
    pred_smd = knncalc(k_knn, smd_distance)

    if pred_em == labels[test_index]:
        correct_em_preds += 1

    if pred_it == labels[test_index]:
        correct_it_preds += 1

    if pred_ed == labels[test_index]:
        correct_ed_preds += 1

    if pred_smd == labels[test_index]:
        correct_smd_preds += 1


total = len(images)
accuracy_entropy = correct_entropy_preds / total
print(f'Accuracy entropy with Leave-One-Out Cross-Validation: {accuracy_entropy * 100:.2f}%')

accuracy_em = correct_em_preds / total
print(f'Accuracy em with Leave-One-Out Cross-Validation: {accuracy_em * 100:.2f}%')

accuracy_it = correct_it_preds / total
print(f'Accuracy it with Leave-One-Out Cross-Validation: {accuracy_it * 100:.2f}%')

accuracy_ed = correct_ed_preds / total
print(f'Accuracy ed with Leave-One-Out Cross-Validation: {accuracy_ed * 100:.2f}%')

accuracy_smd = correct_smd_preds / total
print(f'Accuracy smd with Leave-One-Out Cross-Validation: {accuracy_smd * 100:.2f}%')

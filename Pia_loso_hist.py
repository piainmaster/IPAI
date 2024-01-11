import math
import os

import numpy as np
import cv2
from pathlib import Path
from scipy.stats import wasserstein_distance
from skimage.measure import shannon_entropy
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeaveOneGroupOut

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

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        histBin(img, bin)
        data_PLUS_spoofed.append(["spoofed", img, hist])

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",
                                "spoofed_synthethic_drit",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["003"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, bin)
                    data_PLUS_003.append(["synthethic", img, hist])
        for variant in ["004"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, bin)
                    data_PLUS_004.append(["synthethic", img, hist])

    return data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004


def loadSCUT():
    # Define the path to the dataset
    datasource_path = "dataset/SCUT"

    data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008 = [], [], [], []

    # load dataset SCUT
    for id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                    "041", "045", "049", "053", "057", "061", "065", "069"]:
        p = Path(datasource_path + "/" + "genuine" + "/" + id)
        for filename in p.glob('**/*.bmp'):
            img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
            hist = histBin(img, bin)
            data_SCUT_genuine.append(["genuine", img, hist, id])

    for id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                    "041", "045", "049", "053", "057", "061", "065", "069"]:
        p = Path(datasource_path + "/" + "spoofed" + "/" + id)
        for filename in p.glob('**/*.bmp'):
            img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
            hist = hist = histBin(img, bin)
            data_SCUT_spoofed.append(["spoofed", img, hist, id])

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",
                                "spoofed_synthethic_drit",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["007"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold + "/reference")
                for filename in p.glob('**/*.png'):
                    _, tail = os.path.split(filename)
                    id = tail.split('-', 1)[0]
                    if id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                    "041", "045", "049", "053", "057", "061", "065", "069"]:
                        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                        hist = histBin(img, bin)
                        data_SCUT_007.append(["synthethic", img, hist, id])
        for variant in ["008"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold + "/reference")
                for filename in p.glob('**/*.png'):
                    _, tail = os.path.split(filename)
                    id = tail.split('-', 1)[0]
                    if id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                                   "041", "045", "049", "053", "057", "061", "065", "069"]:
                        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                        hist = histBin(img, bin)
                        data_SCUT_008.append(["synthethic", img, hist, id])

    return data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008


def loadVERA():
    # Define the path to the dataset
    datasource_path = "dataset/IDIAP"

    data_VERA_genuine, data_VERA_spoofed, data_VERA_009 = [], [], []

    # load dataset PLUS
    p = Path(datasource_path + "/" + "genuine")
    for filename in p.glob('**/*.png'):
        _, tail = os.path.split(filename)
        id = tail.split('_', 1)[0]      #todo: only consider images from 001-109 (not -113)
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        hist = histBin(img, bin)
        data_VERA_genuine.append(["genuine", img, hist, id])

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        _, tail = os.path.split(filename)
        id = tail.split('_', 1)[0]  # todo: only consider images from 001-109 (not -113)
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        hist = hist = histBin(img, bin)
        data_VERA_spoofed.append(["spoofed", img, hist, id])

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",
                                "spoofed_synthethic_drit",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["009"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    _, tail = os.path.split(filename)
                    tail_front = tail.split('_', 1)[0]
                    id = tail_front.split('-', 1)[1]
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    hist = histBin(img, bin)
                    data_VERA_009.append(["synthethic", img, hist, id])

    return data_VERA_genuine, data_VERA_spoofed, data_VERA_009



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
    if bins < 257:
        hist = cv2.calcHist([img], [0], None, [bin], [0, bin])
        return hist
    else:
        return "Falscher Bin-Wert"


def intersection_test(hist1, hist2):
    value = np.sum(np.minimum(hist1, hist2))
    return -value


def euclidean_distance_test(hist1, hist2):
    sum = 0
    for i in range(0, bin):
        sum = sum + (hist1[i][0] - hist2[0][i][0]) ** 2
    eu_dist = math.sqrt(sum)
    return eu_dist


def sum_manhattan_distance_test(hist1, hist2):
    sum = 0
    for i in range(0, bin):
        sum = sum + abs(hist1[i][0] - hist2[0][i][0])
    ma_dist = sum
    return ma_dist


# earthmovers distance
def em_dist(img1, img2):
    images = [img.ravel() for img in [img1, img2]]
    em_distance = wasserstein_distance(images[0], images[1])
    return em_distance


# -----------------------------------------------------------------------
# ENTROPY
# -----------------------------------------------------------------------

# calculate Entropy with frames--> takes nr of frames and image path and gives back an array of entropies:
def calculate_entropies(image, num_frames):
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
    # Divide the image into frames
    variances = []
    size = image.shape
    height, width = size[0], size[1]

    frame_size = width // num_frames

    for i in range(num_frames):
        start_x = i * frame_size
        end_x = (i + 1) * frame_size
        frame = image[:, start_x:end_x]
        variance = np.var(frame)
        variances.append(variance)

    return variances


# Function to calculate patch variances of an image with size patch_x x patch_y with no overlap between the patches
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
#data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004 = loadPLUS()
#data_VERA_genuine, data_VERA_spoofed, data_VERA_009 = loadVERA()
data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007,  data_SCUT_008 = loadSCUT()



# -----------------------------------------------------------------------
# CURRENT DATA
# -----------------------------------------------------------------------

def combine_list_with_genuine(list):
    current_data = data_SCUT_genuine + list
    return current_data

# combine lists data_PLUS_genuine and data_PLUS_003
current_data = []
current_data = combine_list_with_genuine(data_SCUT_008)


# convert data to numpy array
labels_list, images_list, histograms_list, id_list = [], [], [], []
for row in current_data:
    labels_list.append(row[0])
    images_list.append(row[1])
    histograms_list.append(row[2])
    id_list.append(row[3])
method = cv2.INTER_LANCZOS4
features = [cv2.resize(img, (736,192), interpolation=method) for img in images_list]      #Alternative sklearn.preprocessing.StandardScaler

labels = np.array(labels_list)
images = np.array(features)
histograms = np.array(histograms_list)
id = np.array(id_list)



# combine lists data_PLUS_genuine and data_PLUS_spoofed
validation_data = []
validation_data = combine_list_with_genuine(data_SCUT_spoofed)      #todo: nur genuine rein, die Pendant zu spoofed haben

# convert data to numpy array
validation_labels_list, validation_images_list, validation_histograms_list, validation_id_list = [], [], [], []
for row in validation_data:
    validation_labels_list.append(row[0])
    validation_images_list.append(row[1])
    validation_histograms_list.append(row[2])
    validation_id_list.append(row[3])
validation_features = [cv2.resize(img, (736, 192), interpolation=method) for img in validation_images_list]  # Alternative sklearn.preprocessing.StandardScaler
validation_labels = np.array(validation_labels_list)
validation_images = np.array(validation_features)
validation_histograms = np.array(validation_histograms_list)
validation_id = np.array(validation_id_list)


# -----------------------------------------------------------------------
# CALCULATE FEATURE ENTROPY
# -----------------------------------------------------------------------
# calculate feature global entropy
num_frames = 1
entropy_list = []
for img in range(len(images)):
    entropies = calculate_entropies(images[img], num_frames)
    entropy_list.append([labels[img], entropies])

# -----------------------------------------------------------------------
# CALCULATE FEATURE VARIANCE
# -----------------------------------------------------------------------



# -----------------------------------------------------------------------
# LEAVE ONE OUT CROSS VALIDATION
# -----------------------------------------------------------------------

loo = LeaveOneOut()
pred_list = []
correct_entropy_preds = 0


correct_it_preds = 0
correct_ed_preds = 0
correct_smd_preds = 0


for i, (train_index, test_index) in enumerate(loo.split(validation_images)):
    current_id = validation_id[test_index]
    #print(i)
    # -----------------------------------------------------------------------
    # ENTROPY
    # -----------------------------------------------------------------------
    #calculate distances
    val_entropies = calculate_entropies(validation_images[test_index], num_frames=1)
    test_entropy = [validation_labels[test_index][0], val_entropies]

    entropy_distances = []
    for j in range(len(labels)):
        if (current_id != id[j]):
            entropy_distances.append([entropy_list[j][0], abs(entropy_list[j][1][0] - test_entropy[1][0])])     #linalg as second method
    #test_entropy = entropy_list[test_index[0]]

    #prediction for current test image
    k_knn = 3
    pred_entropy = knncalc(k_knn, entropy_distances)
    if pred_entropy == 'genuine' and test_entropy[0] == 'genuine' or pred_entropy == 'synthethic' and test_entropy[0] == 'spoofed':
        correct_entropy_preds += 1

    # -----------------------------------------------------------------------
    # VARIANCE
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # HISTOGRAM
    # -----------------------------------------------------------------------


    # intersection distance (runs fast)
    it_distance = []
    for j in range(len(labels)):
        if (current_id != id[j]):
            it = intersection_test(histograms[j], validation_histograms[test_index])
            it_distance.append([labels[j], it])

    # euclidian distance
    ed_distance = []
    for j in range(len(labels)):
        if (current_id != id[j]):
            ed = euclidean_distance_test(histograms[j], validation_histograms[test_index])
            ed_distance.append([labels[j], ed])

    # sum of manhattan distances
    smd_distance = []
    for j in range(len(labels)):
        if (current_id != id[j]):
            smd = sum_manhattan_distance_test(histograms[j], validation_histograms[test_index])
            smd_distance.append([labels[j], smd])

    # prediction for current test image
    k_knn = 3
    # pred_em = knncalc(k_knn, em_distance)
    pred_it = knncalc(k_knn, it_distance)
    pred_ed = knncalc(k_knn, ed_distance)
    pred_smd = knncalc(k_knn, smd_distance)

    # pred_hist_combined = []
    # pred_hist_combined.append([pred_it, pred_ed, pred_smd])
    # pred_hist_combined_value = most_common_value(pred_hist_combined)

    if pred_it == 'genuine' and test_entropy[0] == 'genuine' or pred_it == 'synthethic' and test_entropy[0] == 'spoofed':
        correct_it_preds += 1

    test_entropy = [validation_labels[test_index][0], val_entropies]

    if pred_it == 'genuine' and validation_labels[test_index][0] == 'genuine' or pred_it == 'synthethic' and validation_labels[test_index][0] == 'spoofed':
        correct_it_preds += 1

    if pred_ed == 'genuine' and validation_labels[test_index][0] == 'genuine' or pred_ed == 'synthethic' and validation_labels[test_index][0] == 'spoofed':
        correct_ed_preds += 1

    if pred_smd == 'genuine' and validation_labels[test_index][0] == 'genuine' or pred_smd == 'synthethic' and validation_labels[test_index][0] == 'spoofed':
        correct_smd_preds += 1




total = len(validation_labels)
print(f'Total number of samples: {str(total)}')

print(f'Accuracy entropy: {correct_entropy_preds / total * 100:.2f}%')

print(f'Classification of data_PLUS_genuine and data_PLUS_003 \nVariances with knn {k_knn}')


accuracy_it = correct_it_preds / total
print(f'Accuracy it: {accuracy_it * 100:.2f}%')

accuracy_ed = correct_ed_preds / total
print(f'Accuracy ed: {accuracy_ed * 100:.2f}%')

accuracy_smd = correct_smd_preds / total
print(f'Accuracy smd: {accuracy_smd * 100:.2f}%')

accuracy_hist_combined = correct_hist_combined_preds / total
print(f"Accuracy combined of hist: {accuracy_hist_combined*100:.2f}%")

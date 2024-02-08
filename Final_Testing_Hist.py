import math
import os

import numpy as np
import cv2
from pathlib import Path
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
#from skimage.measure import shannon_entropy
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeaveOneGroupOut

method = cv2.INTER_LANCZOS4

def loadPLUS():
    # Define the path to the dataset
    datasource_path = "dataset/PLUS"

    data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004 = [], [], [], []

    # load dataset PLUS
    p = Path(datasource_path + "/" + "genuine")
    for filename in p.glob('**/*.png'):
        _, tail = str(filename).rsplit("Laser_PALMAR_", 1)
        id = tail.split('_', 1)[0]
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (736, 192), interpolation=method)
        hist = histBin(img, bin)
        data_PLUS_genuine.append(["genuine", img, hist, id])

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        _, tail = str(filename).rsplit("Laser_PALMAR_", 1)
        id = tail.split('_', 1)[0]
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (736, 192), interpolation=method)
        histBin(img, bin)
        data_PLUS_spoofed.append(["spoofed", img, hist, id])

    for synthethic_category in ["spoofed_synthethic_cyclegan",
                                "spoofed_synthethic_distancegan",
                                "spoofed_synthethic_drit",
                                "spoofed_synthethic_stargan-v2"]:
        for variant in ["003"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    _, tail = str(filename).rsplit("Laser_PALMAR_", 1)
                    id = tail.split('_', 1)[0]
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (736, 192), interpolation=method)
                    hist = histBin(img, bin)
                    data_PLUS_003.append([synthethic_category, img, hist, id])
        for variant in ["004"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold)
                for filename in p.glob('**/*.png'):
                    _, tail = str(filename).rsplit("Laser_PALMAR_", 1)
                    id = tail.split('_', 1)[0]
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (736, 192), interpolation=method)
                    hist = histBin(img, bin)
                    data_PLUS_004.append([synthethic_category, img, hist, id])

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
            img = cv2.rotate(cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE), cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (639, 287), interpolation=method)
            hist = histBin(img, bin)
            data_SCUT_genuine.append(["genuine", img, hist, id])

    for id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                    "041", "045", "049", "053", "057", "061", "065", "069"]:
        p = Path(datasource_path + "/" + "spoofed" + "/" + id)
        for filename in p.glob('**/*.bmp'):
            img = cv2.rotate(cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE), cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (639, 287), interpolation=method)
            hist = histBin(img, bin)
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
                        img = cv2.resize(img, (639, 287), interpolation=method)
                        hist = histBin(img, bin)
                        hist = histBin(img, bin)
                        data_SCUT_007.append([synthethic_category, img, hist, id])
                for filename in p.glob('**/*.jpg'):
                    _, tail = os.path.split(filename)
                    id = tail.split('-', 1)[0]
                    if id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                              "041", "045", "049", "053", "057", "061", "065", "069"]:
                        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (639, 287), interpolation=method)
                        hist = histBin(img, bin)
                        hist = histBin(img, bin)
                        data_SCUT_007.append([synthethic_category, img, hist, id])
        for variant in ["008"]:
            for fold in ["1", "2", "3", "4", "5"]:
                p = Path(datasource_path + "/" + synthethic_category + "/" + variant + "/" + fold + "/reference")
                for filename in p.glob('**/*.png'):
                    _, tail = os.path.split(filename)
                    id = tail.split('-', 1)[0]
                    if id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                                   "041", "045", "049", "053", "057", "061", "065", "069"]:
                        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (639, 287), interpolation=method)
                        hist = histBin(img, bin)
                        data_SCUT_008.append([synthethic_category, img, hist, id])
                for filename in p.glob('**/*.jpg'):
                    _, tail = os.path.split(filename)
                    id = tail.split('-', 1)[0]
                    if id in ["001", "005", "009", "013", "017", "021", "025", "029", "033", "037",
                              "041", "045", "049", "053", "057", "061", "065", "069"]:
                        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (639, 287), interpolation=method)
                        hist = histBin(img, bin)
                        data_SCUT_008.append([synthethic_category, img, hist, id])

    return data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008


def loadVERA():
    # Define the path to the dataset
    datasource_path = "dataset/IDIAP"

    data_VERA_genuine, data_VERA_spoofed, data_VERA_009 = [], [], []

    # load dataset PLUS
    p = Path(datasource_path + "/" + "genuine")
    for filename in p.glob('**/*.png'):
        _, tail = os.path.split(filename)
        id = tail.split('_', 1)[0]  # todo: only consider images from 001-109 (not -113)
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (664, 248), interpolation=method)
        hist = histBin(img, bin)
        data_VERA_genuine.append(["genuine", img, hist, id])

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        _, tail = os.path.split(filename)
        id = tail.split('_', 1)[0]  # todo: only consider images from 001-109 (not -113)
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (664, 248), interpolation=method)
        hist = histBin(img, bin)
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
                    img = cv2.resize(img, (664, 248), interpolation=method)
                    hist = histBin(img, bin)
                    data_VERA_009.append([synthethic_category, img, hist, id])

                for filename in p.glob('**/*.jpg'):
                    _, tail = os.path.split(filename)
                    tail_front = tail.split('_', 1)[0]
                    id = tail_front.split('-', 1)[1]
                    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (664, 248), interpolation=method)
                    hist = histBin(img, bin)
                    data_VERA_009.append([synthethic_category, img, hist, id])

    return data_VERA_genuine, data_VERA_spoofed, data_VERA_009


from skimage.measure import shannon_entropy
import scipy


def combine_list_with_genuine(list):
    current_data = data_genuine + list
    return current_data


# -----------------------------------------------------------------------
# KNN
# -----------------------------------------------------------------------
def knncalc(k, feature_list):
    knn_list = list()
    feature_list.sort(key=lambda tup: tup[1])  # nach den Werten in der zweiten Spalte sortieren (kng)
    for i in range(k):
        knn_list.append(feature_list[i][0])  # es werden "spoofed" und "genuine" eingetragen
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
        hist = cv2.calcHist([img], [0], None, [bin], [0, 256])
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
# METHODEN-AUFRUFE
# -----------------------------------------------------------------------
method = cv2.INTER_LANCZOS4
bin = 6

data_genuine, data_spoofed, data_PLUS_003, data_PLUS_004 = loadPLUS()
#data_genuine, data_spoofed, data_VERA_009 = loadVERA()
#data_genuine, data_spoofed, data_SCUT_007,  data_SCUT_008 = loadSCUT()

# to be changed:
generation_variant = data_PLUS_003
generation_method = "spoofed_synthethic_cyclegan" #"spoofed_synthethic_cyclegan"  #"spoofed_synthethic_distancegan"#"spoofed_synthethic_drit"  #"spoofed_synthethic_stargan-v2"

data_synthetic = []
for row in generation_variant:
    if row[0] == generation_method:
        row[0] = "synthetic"
        data_synthetic.append(row)

if len(data_synthetic) < len(data_genuine):
    data_genuine = data_genuine[:(len(data_synthetic))]
    data_spoofed = data_spoofed[:(len(data_synthetic))]


# -----------------------------------------------------------------------
# CURRENT TRAIN DATA
# -----------------------------------------------------------------------
# combine genuine data with synthetic data
current_data = []
current_data = combine_list_with_genuine(data_synthetic)

# convert data to numpy array
labels_list, images_list, histograms_list, id_list = [], [], [], []
for row in current_data:
    labels_list.append(row[0])
    images_list.append(row[1])
    histograms_list.append(row[2])
    id_list.append(row[3])

labels = np.array(labels_list)
images = np.array(images_list)
histograms = np.array(histograms_list)
id = np.array(id_list)

# -----------------------------------------------------------------------
# CURRENT TEST DATA
# -----------------------------------------------------------------------
# combine lists data_PLUS_genuine and data_PLUS_spoofed
validation_data = []
validation_data = combine_list_with_genuine(data_spoofed)

# convert data to numpy array
validation_labels_list, validation_images_list, validation_histograms_list, validation_id_list = [], [], [], []
for row in validation_data:
    validation_labels_list.append(row[0])
    validation_images_list.append(row[1])
    validation_histograms_list.append(row[2])
    validation_id_list.append(row[3])

validation_labels = np.array(validation_labels_list)
validation_images = np.array(validation_images_list)
validation_histograms = np.array(validation_histograms_list)
validation_id = np.array(validation_id_list)

loo = LeaveOneOut()
pred_list = []

# -----------------------------------------------------------------------
# CALCULATE FEATURE ENTROPY
# -----------------------------------------------------------------------

k_knn = 3


correct_it_preds = 0

for i, (train_index, test_index) in enumerate(loo.split(validation_images)):
    current_id = validation_id[test_index]

    it_distance = []
    for j in range(len(labels)):
        it = intersection_test(histograms[j], validation_histograms[test_index])
        it_distance.append([labels[j], it])

    pred_it = knncalc(k_knn, it_distance)
    if pred_it == 'genuine' and validation_labels[test_index][0] == 'genuine' or pred_it != 'genuine' and \
            validation_labels[test_index][0] != 'genuine':
        correct_it_preds += 1
    # prediction for current test image


# -----------------------------------------------------------------------
# OUTPUT RESULTS
# -----------------------------------------------------------------------
print(f'Leave One Out')
total = len(validation_labels)
print(f'Total number of samples: {str(total)}')
print(f'Classification results \nEntropies with knn {k_knn}')

print(f'Accuracy entropy with 50x50 patches: {correct_it_preds / total * 100:.3f}%')

# -----------------------------------------------------------------------
# LEAVE ONE SUBJECT OUT
# -----------------------------------------------------------------------
knn = 5

correct_entropy50x50_preds = 0

for i, (train_index, test_index) in enumerate(loo.split(validation_images)):
    current_id = validation_id[test_index]


    test_entropy50x50 = [validation_labels[test_index][0],
                          calculate_patch_entropies(validation_images[test_index][0], 50, 50)]
    entropy50x50_distances = []

    for j in range(len(labels)):
        if (current_id != id[j]):
            entropy50x50_distances.append([entropy_list50x50[j][0], mean_absolute_difference(entropy_list50x50[j][1], test_entropy50x50[1])])

    # prediction for current test image
    if knncalc(k_knn, entropy50x50_distances) == 'genuine' and test_entropy50x50[0] == 'genuine' or knncalc(k_knn, entropy50x50_distances) != 'genuine' and \
            test_entropy50x50[0] != 'genuine': correct_entropy50x50_preds += 1

# -----------------------------------------------------------------------
# OUTPUT RESULTS
# -----------------------------------------------------------------------
print(f'Leave One Subject Out')
total = len(validation_labels)
print(f'Total number of samples: {str(total)}')
print(f'Classification results \nEntropies with knn {k_knn}')

print(f'Accuracy entropy with 50x50 patches: {correct_entropy50x50_preds / total * 100:.3f}%')
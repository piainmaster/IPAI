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


# Function to calculate patch entropies of an image with size patch_x x patch_y with no overlap between the patches
def calculate_patch_entropies(image, patch_x, patch_y):
    patch_size = (patch_x, patch_y)
    entropies = []
    h, w = image.shape
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            entropies.append(shannon_entropy(patch))
    return entropies


# -----------------------------------------------------------------------
# VARIANCE
# -----------------------------------------------------------------------
def mean_absolute_difference(a, b):
    if len(a) != len(b):
        raise ValueError("Input lists must have the same length")

    return sum(abs(a_i - b_i) for a_i, b_i in zip(a, b)) / len(a)


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
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            variances.append(np.var(patch))
    return variances


# -----------------------------------------------------------------------
# METHODEN-AUFRUFE
# -----------------------------------------------------------------------
method = cv2.INTER_LANCZOS4
bin = 6

#data_genuine, data_spoofed, data_PLUS_003, data_PLUS_004 = loadPLUS()
data_genuine, data_spoofed, data_VERA_009 = loadVERA()
#data_genuine, data_spoofed, data_SCUT_007,  data_SCUT_008 = loadSCUT()

# to be changed:
generation_variant = data_VERA_009
generation_method = "spoofed_synthethic_cyclegan" #"spoofed_synthethic_distancegan"  #"spoofed_synthethic_distancegan"#"spoofed_synthethic_drit"  #"spoofed_synthethic_stargan-v2"

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
# CALCULATE FEATURE VARIANCE
# -----------------------------------------------------------------------

# calculate feature variance
variance_list1 = []
variance_list10 = []
variance_list20 = []
variance_list30 = []
variance_list40 = []
variance_list50 = []
variance_list100 = []

#note: these patches divide image in vertical stripes
for img in range(len(images)):
    #global variance
    variances1 = calculate_variances(images[img], 1)
    variance_list1.append([labels[img], variances1])
    #variance with 10 patches
    variances10 = calculate_variances(images[img], 10)
    variance_list10.append([labels[img], variances10])
    #variance with 20 patches
    variances20 = calculate_variances(images[img], 20)
    variance_list20.append([labels[img], variances20])
    #variance with 30 patches
    variances30 = calculate_variances(images[img], 30)
    variance_list30.append([labels[img], variances30])
    #variance with 40 patches
    variances40 = calculate_variances(images[img], 40)
    variance_list40.append([labels[img], variances40])
    #variance with 50 patches
    variances50 = calculate_variances(images[img], 50)
    variance_list50.append([labels[img], variances50])
    #variance with 100 patches
    variances100 = calculate_variances(images[img], 100)
    variance_list100.append([labels[img], variances100])



variance_list10x10 = []
variance_list20x20 = []
variance_list30x30 = []
variance_list40x40 = []
variance_list50x50 = []
variance_list70x70 = []
variance_list100x100 = []
variance_list100x200 = []

#note: these patches divide in a x b sized patches
for img in range(len(images)):
    #variance with 10x10 patches
    variances10x10 = calculate_patch_variances(images[img], 10, 10)
    variance_list10x10.append([labels[img], variances10x10])
    #variance with 20x20 patches
    variances20x20 = calculate_patch_variances(images[img], 20, 20)
    variance_list20x20.append([labels[img], variances20x20])
    #variance with 30x30 patches
    variances30x30 = calculate_patch_variances(images[img], 30, 30)
    variance_list30x30.append([labels[img], variances30x30])
    #variance with 40x40 patches
    variances40x40 = calculate_patch_variances(images[img], 40, 40)
    variance_list40x40.append([labels[img], variances40x40])
    #variance with 50x50 patches
    variances50x50 = calculate_patch_variances(images[img], 50, 50)
    variance_list50x50.append([labels[img], variances50x50])
    # variance with 70x70 patches
    variances70x70 = calculate_patch_variances(images[img], 70, 70)
    variance_list70x70.append([labels[img], variances70x70])
    #variance with 100x100 patches
    variances100x100 = calculate_patch_variances(images[img], 100, 100)
    variance_list100x100.append([labels[img], variances100x100])
    #variance with 100x200 patches
    variances100x200 = calculate_patch_variances(images[img], 100, 200)
    variance_list100x200.append([labels[img], variances100x200])

for k_knn in [3, 5, 7, 9]:

    correct_variance1_preds = 0
    correct_variance10_preds = 0
    correct_variance20_preds = 0
    correct_variance30_preds = 0
    correct_variance40_preds = 0
    correct_variance50_preds = 0
    correct_variance100_preds = 0

    correct_variance10x10_preds = 0
    correct_variance20x20_preds = 0
    correct_variance30x30_preds = 0
    correct_variance40x40_preds = 0
    correct_variance50x50_preds = 0
    correct_variance70x70_preds = 0
    correct_variance100x100_preds = 0
    correct_variance100x200_preds = 0

    for i, (train_index, test_index) in enumerate(loo.split(validation_images)):
        current_id = validation_id[test_index]

        test_variance1 = [validation_labels[test_index][0],
                          calculate_variances(validation_images[test_index], num_frames=1)]
        variance1_distances = []
        test_variance10 = [validation_labels[test_index][0], calculate_variances(validation_images[test_index][0], 10)]
        variance10_distances = []
        test_variance20 = [validation_labels[test_index][0], calculate_variances(validation_images[test_index][0], 20)]
        variance20_distances = []
        test_variance30 = [validation_labels[test_index][0], calculate_variances(validation_images[test_index][0], 30)]
        variance30_distances = []
        test_variance40 = [validation_labels[test_index][0], calculate_variances(validation_images[test_index][0], 40)]
        variance40_distances = []
        test_variance50 = [validation_labels[test_index][0], calculate_variances(validation_images[test_index][0], 50)]
        variance50_distances = []
        test_variance100 = [validation_labels[test_index][0],
                            calculate_variances(validation_images[test_index][0], 100)]
        variance100_distances = []

        test_variance10x10 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 10, 10)]
        variance10x10_distances = []
        test_variance20x20 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 20, 20)]
        variance20x20_distances = []
        test_variance30x30 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 30, 30)]
        variance30x30_distances = []
        test_variance40x40 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 40, 40)]
        variance40x40_distances = []
        test_variance50x50 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 50, 50)]
        variance50x50_distances = []
        test_variance70x70 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 70, 70)]
        variance70x70_distances = []
        test_variance100x100 = [validation_labels[test_index][0],
                                calculate_patch_variances(validation_images[test_index][0], 100, 100)]
        variance100x100_distances = []
        test_variance100x200 = [validation_labels[test_index][0],
                                calculate_patch_variances(validation_images[test_index][0], 100, 200)]
        variance100x200_distances = []

        for j in range(len(labels)):
            variance1_distances.append([variance_list1[j][0], abs(variance_list1[j][1][0] - test_variance1[1][0])])
            variance10_distances.append(
                [variance_list10[j][0], mean_absolute_difference(variance_list10[j][1], test_variance10[1])])
            variance20_distances.append(
                [variance_list20[j][0], mean_absolute_difference(variance_list20[j][1], test_variance20[1])])
            variance30_distances.append(
                [variance_list30[j][0], mean_absolute_difference(variance_list30[j][1], test_variance30[1])])
            variance40_distances.append(
                [variance_list40[j][0], mean_absolute_difference(variance_list40[j][1], test_variance40[1])])
            variance50_distances.append(
                [variance_list50[j][0], mean_absolute_difference(variance_list50[j][1], test_variance50[1])])
            variance100_distances.append(
                [variance_list100[j][0], mean_absolute_difference(variance_list100[j][1], test_variance100[1])])

            variance10x10_distances.append(
                [variance_list10x10[j][0], mean_absolute_difference(variance_list10x10[j][1], test_variance10x10[1])])
            variance20x20_distances.append(
                [variance_list20x20[j][0], mean_absolute_difference(variance_list20x20[j][1], test_variance20x20[1])])
            variance30x30_distances.append(
                [variance_list30x30[j][0], mean_absolute_difference(variance_list30x30[j][1], test_variance30x30[1])])
            variance40x40_distances.append(
                [variance_list40x40[j][0], mean_absolute_difference(variance_list40x40[j][1], test_variance40x40[1])])
            variance50x50_distances.append(
                [variance_list50x50[j][0], mean_absolute_difference(variance_list50x50[j][1], test_variance50x50[1])])
            variance70x70_distances.append(
                [variance_list70x70[j][0], mean_absolute_difference(variance_list70x70[j][1], test_variance70x70[1])])
            variance100x100_distances.append([variance_list100x100[j][0],
                                              mean_absolute_difference(variance_list100x100[j][1],
                                                                       test_variance100x100[1])])
            variance100x200_distances.append([variance_list100x200[j][0],
                                              mean_absolute_difference(variance_list100x200[j][1],
                                                                       test_variance100x200[1])])

        # prediction for current test image
        if (knncalc(k_knn, variance1_distances) == 'genuine' and test_variance1[0] == 'genuine') or (
                knncalc(k_knn, variance1_distances) != 'genuine' and test_variance1[
            0] != 'genuine'): correct_variance1_preds += 1
        if (knncalc(k_knn, variance10_distances) == 'genuine' and test_variance10[0] == 'genuine') or (
                knncalc(k_knn, variance10_distances) != 'genuine' and test_variance10[
            0] != 'genuine'): correct_variance10_preds += 1
        if knncalc(k_knn, variance20_distances) == 'genuine' and test_variance20[0] == 'genuine' or knncalc(k_knn,
                                                                                                            variance20_distances) != 'genuine' and \
                test_variance20[0] != 'genuine': correct_variance20_preds += 1
        if knncalc(k_knn, variance30_distances) == 'genuine' and test_variance30[0] == 'genuine' or knncalc(k_knn,
                                                                                                            variance30_distances) != 'genuine' and \
                test_variance30[0] != 'genuine': correct_variance30_preds += 1
        if knncalc(k_knn, variance40_distances) == 'genuine' and test_variance40[0] == 'genuine' or knncalc(k_knn,
                                                                                                            variance40_distances) != 'genuine' and \
                test_variance40[0] != 'genuine': correct_variance40_preds += 1
        if knncalc(k_knn, variance50_distances) == 'genuine' and test_variance50[0] == 'genuine' or knncalc(k_knn,
                                                                                                            variance50_distances) != 'genuine' and \
                test_variance50[0] != 'genuine': correct_variance50_preds += 1
        if knncalc(k_knn, variance100_distances) == 'genuine' and test_variance100[0] == 'genuine' or knncalc(k_knn,
                                                                                                              variance100_distances) != 'genuine' and \
                test_variance100[0] != 'genuine': correct_variance100_preds += 1

        if (knncalc(k_knn, variance10x10_distances) == 'genuine' and test_variance10x10[0] == 'genuine') or (
                knncalc(k_knn, variance10x10_distances) != 'genuine' and test_variance10x10[
            0] != 'genuine'): correct_variance10x10_preds += 1
        if knncalc(k_knn, variance20x20_distances) == 'genuine' and test_variance20x20[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance20x20_distances) != 'genuine' and \
                test_variance20x20[0] != 'genuine': correct_variance20x20_preds += 1
        if knncalc(k_knn, variance30x30_distances) == 'genuine' and test_variance30x30[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance30x30_distances) != 'genuine' and \
                test_variance30x30[0] != 'genuine': correct_variance30x30_preds += 1
        if knncalc(k_knn, variance40x40_distances) == 'genuine' and test_variance40x40[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance40x40_distances) != 'genuine' and \
                test_variance40x40[0] != 'genuine': correct_variance40x40_preds += 1
        if knncalc(k_knn, variance50x50_distances) == 'genuine' and test_variance50x50[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance50x50_distances) != 'genuine' and \
                test_variance50x50[0] != 'genuine': correct_variance50x50_preds += 1
        if knncalc(k_knn, variance70x70_distances) == 'genuine' and test_variance70x70[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance70x70_distances) != 'genuine' and \
                test_variance70x70[0] != 'genuine': correct_variance70x70_preds += 1
        if knncalc(k_knn, variance100x100_distances) == 'genuine' and test_variance100x100[0] == 'genuine' or knncalc(
            k_knn, variance100x100_distances) != 'genuine' and test_variance100x100[
            0] != 'genuine': correct_variance100x100_preds += 1
        if knncalc(k_knn, variance100x200_distances) == 'genuine' and test_variance100x200[0] == 'genuine' or knncalc(
            k_knn, variance100x200_distances) != 'genuine' and test_variance100x200[
            0] != 'genuine': correct_variance100x200_preds += 1

    # -----------------------------------------------------------------------
    # OUTPUT RESULTS
    # -----------------------------------------------------------------------
    print(f'Leave One Out')
    total = len(validation_labels)
    print(f'Total number of samples: {str(total)}')
    print(f'Classification results \nVariances with knn {k_knn}')

    print(f'Accuracy global variance: {correct_variance1_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 10 patches: {correct_variance10_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 20 patches: {correct_variance20_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 30 patches: {correct_variance30_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 40 patches: {correct_variance40_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 50 patches: {correct_variance50_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 100 patches: {correct_variance100_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 10x10 patches: {correct_variance10x10_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 20x20 patches: {correct_variance20x20_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 30x30 patches: {correct_variance30x30_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 40x40 patches: {correct_variance40x40_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 50x50 patches: {correct_variance50x50_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 70x70 patches: {correct_variance70x70_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 100x100 patches: {correct_variance100x100_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 100x200 patches: {correct_variance100x200_preds / total * 100:.3f}%')

for k_knn in [3, 5, 7, 9]:

    correct_variance1_preds = 0
    correct_variance10_preds = 0
    correct_variance20_preds = 0
    correct_variance30_preds = 0
    correct_variance40_preds = 0
    correct_variance50_preds = 0
    correct_variance100_preds = 0
    correct_variance10x10_preds = 0
    correct_variance20x20_preds = 0
    correct_variance30x30_preds = 0
    correct_variance40x40_preds = 0
    correct_variance50x50_preds = 0
    correct_variance70x70_preds = 0
    correct_variance100x100_preds = 0
    correct_variance100x200_preds = 0

    for i, (train_index, test_index) in enumerate(loo.split(validation_images)):
        current_id = validation_id[test_index]

        test_variance1 = [validation_labels[test_index][0],
                          calculate_variances(validation_images[test_index], num_frames=1)]
        variance1_distances = []
        test_variance10 = [validation_labels[test_index][0],
                           calculate_variances(validation_images[test_index], num_frames=10)]
        variance10_distances = []
        test_variance20 = [validation_labels[test_index][0],
                           calculate_variances(validation_images[test_index], num_frames=20)]
        variance20_distances = []
        test_variance30 = [validation_labels[test_index][0],
                           calculate_variances(validation_images[test_index], num_frames=30)]
        variance30_distances = []
        test_variance40 = [validation_labels[test_index][0],
                           calculate_variances(validation_images[test_index], num_frames=40)]
        variance40_distances = []
        test_variance50 = [validation_labels[test_index][0],
                           calculate_variances(validation_images[test_index], num_frames=50)]
        variance50_distances = []
        test_variance100 = [validation_labels[test_index][0],
                            calculate_variances(validation_images[test_index], num_frames=100)]
        variance100_distances = []

        test_variance10x10 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 10, 10)]
        variance10x10_distances = []
        test_variance20x20 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 20, 20)]
        variance20x20_distances = []
        test_variance30x30 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 30, 30)]
        variance30x30_distances = []
        test_variance40x40 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 40, 40)]
        variance40x40_distances = []
        test_variance50x50 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 50, 50)]
        variance50x50_distances = []
        test_variance70x70 = [validation_labels[test_index][0],
                              calculate_patch_variances(validation_images[test_index][0], 70, 70)]
        variance70x70_distances = []
        test_variance100x100 = [validation_labels[test_index][0],
                                calculate_patch_variances(validation_images[test_index][0], 100, 100)]
        variance100x100_distances = []
        test_variance100x200 = [validation_labels[test_index][0],
                                calculate_patch_variances(validation_images[test_index][0], 100, 200)]
        variance100x200_distances = []

        for j in range(len(labels)):
            if (current_id != id[j]):
                variance1_distances.append([variance_list1[j][0], abs(variance_list1[j][1][0] - test_variance1[1][0])])
                variance10_distances.append(
                    [variance_list10[j][0], mean_absolute_difference(variance_list10[j][1], test_variance10[1])])
                variance20_distances.append(
                    [variance_list20[j][0], mean_absolute_difference(variance_list20[j][1], test_variance20[1])])
                variance30_distances.append(
                    [variance_list30[j][0], mean_absolute_difference(variance_list30[j][1], test_variance30[1])])
                variance40_distances.append(
                    [variance_list40[j][0], mean_absolute_difference(variance_list40[j][1], test_variance40[1])])
                variance50_distances.append(
                    [variance_list50[j][0], mean_absolute_difference(variance_list50[j][1], test_variance50[1])])
                variance100_distances.append(
                    [variance_list100[j][0], mean_absolute_difference(variance_list100[j][1], test_variance100[1])])

                variance10x10_distances.append([variance_list10x10[j][0],
                                                mean_absolute_difference(variance_list10x10[j][1],
                                                                         test_variance10x10[1])])
                variance20x20_distances.append([variance_list20x20[j][0],
                                                mean_absolute_difference(variance_list20x20[j][1],
                                                                         test_variance20x20[1])])
                variance30x30_distances.append([variance_list30x30[j][0],
                                                mean_absolute_difference(variance_list30x30[j][1],
                                                                         test_variance30x30[1])])
                variance40x40_distances.append([variance_list40x40[j][0],
                                                mean_absolute_difference(variance_list40x40[j][1],
                                                                         test_variance40x40[1])])
                variance50x50_distances.append([variance_list50x50[j][0],
                                                mean_absolute_difference(variance_list50x50[j][1],
                                                                         test_variance50x50[1])])
                variance70x70_distances.append([variance_list70x70[j][0],
                                                mean_absolute_difference(variance_list70x70[j][1],
                                                                         test_variance70x70[1])])
                variance100x100_distances.append([variance_list100x100[j][0],
                                                  mean_absolute_difference(variance_list100x100[j][1],
                                                                           test_variance100x100[1])])
                variance100x200_distances.append([variance_list100x200[j][0],
                                                  mean_absolute_difference(variance_list100x200[j][1],
                                                                           test_variance100x200[1])])

        # prediction for current test image
        if (knncalc(k_knn, variance1_distances) == 'genuine' and test_variance1[0] == 'genuine') or (
                knncalc(k_knn, variance1_distances) != 'genuine' and test_variance1[
            0] != 'genuine'): correct_variance1_preds += 1
        if (knncalc(k_knn, variance10_distances) == 'genuine' and test_variance10[0] == 'genuine') or (
                knncalc(k_knn, variance10_distances) != 'genuine' and test_variance10[
            0] != 'genuine'): correct_variance10_preds += 1
        if (knncalc(k_knn, variance20_distances) == 'genuine' and test_variance20[0] == 'genuine') or (
                knncalc(k_knn, variance20_distances) != 'genuine' and test_variance20[
            0] != 'genuine'): correct_variance20_preds += 1
        if (knncalc(k_knn, variance30_distances) == 'genuine' and test_variance30[0] == 'genuine') or (
                knncalc(k_knn, variance30_distances) != 'genuine' and test_variance30[
            0] != 'genuine'): correct_variance30_preds += 1
        if (knncalc(k_knn, variance40_distances) == 'genuine' and test_variance40[0] == 'genuine') or (
                knncalc(k_knn, variance40_distances) != 'genuine' and test_variance40[
            0] != 'genuine'): correct_variance40_preds += 1
        if (knncalc(k_knn, variance50_distances) == 'genuine' and test_variance50[0] == 'genuine') or (
                knncalc(k_knn, variance50_distances) != 'genuine' and test_variance50[
            0] != 'genuine'): correct_variance50_preds += 1
        if (knncalc(k_knn, variance100_distances) == 'genuine' and test_variance100[0] == 'genuine') or (
                knncalc(k_knn, variance100_distances) != 'genuine' and test_variance100[
            0] != 'genuine'): correct_variance100_preds += 1

        if (knncalc(k_knn, variance10x10_distances) == 'genuine' and test_variance10x10[0] == 'genuine') or (
                knncalc(k_knn, variance10x10_distances) != 'genuine' and test_variance10x10[
            0] != 'genuine'): correct_variance10x10_preds += 1
        if knncalc(k_knn, variance20x20_distances) == 'genuine' and test_variance20x20[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance20x20_distances) != 'genuine' and \
                test_variance20x20[0] != 'genuine': correct_variance20x20_preds += 1
        if knncalc(k_knn, variance30x30_distances) == 'genuine' and test_variance30x30[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance30x30_distances) != 'genuine' and \
                test_variance30x30[0] != 'genuine': correct_variance30x30_preds += 1
        if knncalc(k_knn, variance40x40_distances) == 'genuine' and test_variance40x40[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance40x40_distances) != 'genuine' and \
                test_variance40x40[0] != 'genuine': correct_variance40x40_preds += 1
        if knncalc(k_knn, variance50x50_distances) == 'genuine' and test_variance50x50[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance50x50_distances) != 'genuine' and \
                test_variance50x50[0] != 'genuine': correct_variance50x50_preds += 1
        if knncalc(k_knn, variance70x70_distances) == 'genuine' and test_variance70x70[0] == 'genuine' or knncalc(k_knn,
                                                                                                                  variance70x70_distances) != 'genuine' and \
                test_variance70x70[0] != 'genuine': correct_variance70x70_preds += 1
        if knncalc(k_knn, variance100x100_distances) == 'genuine' and test_variance100x100[0] == 'genuine' or knncalc(
            k_knn, variance100x100_distances) != 'genuine' and test_variance100x100[
            0] != 'genuine': correct_variance100x100_preds += 1
        if knncalc(k_knn, variance100x200_distances) == 'genuine' and test_variance100x200[0] == 'genuine' or knncalc(
            k_knn, variance100x200_distances) != 'genuine' and test_variance100x200[
            0] != 'genuine': correct_variance100x200_preds += 1

    # -----------------------------------------------------------------------
    # OUTPUT RESULTS
    # -----------------------------------------------------------------------
    print(f'Leave One Subject Out')
    total = len(validation_labels)
    print(f'Total number of samples: {str(total)}')
    print(f'Classification results \nVariances with knn {k_knn}')

    print(f'Accuracy global variance: {correct_variance1_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 10 patches: {correct_variance10_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 20 patches: {correct_variance20_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 30 patches: {correct_variance30_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 40 patches: {correct_variance40_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 50 patches: {correct_variance50_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 100 patches: {correct_variance100_preds / total * 100:.3f}%')

    print(f'Accuracy variance with 10x10 patches: {correct_variance10x10_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 20x20 patches: {correct_variance20x20_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 30x30 patches: {correct_variance30x30_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 40x40 patches: {correct_variance40x40_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 50x50 patches: {correct_variance50x50_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 70x70 patches: {correct_variance70x70_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 100x100 patches: {correct_variance100x100_preds / total * 100:.3f}%')
    print(f'Accuracy variance with 100x200 patches: {correct_variance100x200_preds / total * 100:.3f}%')
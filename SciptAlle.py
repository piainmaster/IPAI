import math
import numpy as np
import cv2
from pathlib import Path
from scipy.stats import wasserstein_distance
from skimage.measure import shannon_entropy
from sklearn.model_selection import LeaveOneOut


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

    p = Path(datasource_path + "/" + "spoofed")
    for filename in p.glob('**/*.png'):
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        histBin(img, bin)
        data_PLUS_spoofed.append(["spoofed", img, hist])

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
        hist = cv2.calcHist([img], [0], None, [bin], [0,bin])
        return hist
    else:
        return "Falscher Bin-Wert"

def intersection_test(hist1, hist2):
    value = np.sum(np.minimum(hist1, hist2))
    return -value

def euclidean_distance_test(hist1, hist2):
    sum = 0
    for i in range (0,bin):
        sum = sum + (hist1[i][0]-hist2[i][0])**2
    eu_dist = math.sqrt(sum)
    return eu_dist

def sum_manhattan_distance_test(hist1, hist2):
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
data_PLUS_genuine, data_PLUS_spoofed, data_PLUS_003, data_PLUS_004 = loadPLUS()
#data_SCUT_genuine, data_SCUT_spoofed, data_SCUT_007, data_SCUT_008 = loadSCUT()




# -----------------------------------------------------------------------
# CURRENT DATA
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
method = cv2.INTER_LANCZOS4
features = [cv2.resize(img, (736,192), interpolation=method) for img in images_list]      #Alternative sklearn.preprocessing.StandardScaler

labels = np.array(labels_list)
images = np.array(features)
histograms = np.array(histograms_list)



# combine lists data_PLUS_genuine and data_PLUS_spoofed
validation_data = []
validation_data = data_PLUS_spoofed      #todo: nur genuine rein, die Pendant zu spoofed haben

# convert data to numpy array
validation_labels_list, validation_images_list, validation_histograms_list = [], [], []
counter = 0
for row in validation_data:
    validation_labels_list.append(row[0])
    validation_images_list.append(row[1])
    validation_histograms_list.append(row[2])
validation_features = [cv2.resize(img, (736,192), interpolation=method) for img in validation_images_list]      #Alternative sklearn.preprocessing.StandardScaler
validation_labels = np.array(validation_labels_list)
validation_images = np.array(validation_features)
validation_histograms = np.array(validation_histograms_list)


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

# calculate feature variance
variance_list1 = []
variance_list5 = []
variance_list10 = []
"""
variance_list20 = []
variance_list30 = []
variance_list40 = []
variance_list50 = []
variance_list100 = []
"""

#note: these patches divide image in vertical stripes
for img in range(len(images)):
    #global variance
    variances1 = calculate_variances(images[img], 1)
    variance_list1.append([labels[img], variances1])
    # variance with 5 patches
    variances5 = calculate_variances(images[img], 5)
    variance_list5.append([labels[img], variances5])
    #variance with 10 patches
    variances10 = calculate_variances(images[img], 10)
    variance_list10.append([labels[img], variances10])
    """
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
    """

"""
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

"""

# -----------------------------------------------------------------------
# LEAVE ONE OUT CROSS VALIDATION
# -----------------------------------------------------------------------

loo = LeaveOneOut()
pred_list = []
correct_entropy_preds = 0

correct_variance1_preds = 0
correct_variance5_preds = 0
correct_variance10_preds = 0
"""
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
"""

"""
correct_em_preds = 0
correct_it_preds = 0
correct_ed_preds = 0
correct_smd_preds = 0
correct_hist_combined_preds = 0
"""

for i, (train_index, test_index) in enumerate(loo.split(validation_images)):
    # -----------------------------------------------------------------------
    # ENTROPY
    # -----------------------------------------------------------------------
    #calculate distances
    val_entropies = calculate_entropies(validation_images[test_index], num_frames=1)
    test_entropy = [validation_labels[test_index][0], val_entropies]

    entropy_distances = []
    for j in range(len(labels)):
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

    #global variance
    test_variance1 = variance_list1[test_index[0]]
    variance1_distances = []
    #test_variance10 = variance_list10[test_index[0]]
    val_variances10 = calculate_variances(validation_images[test_index], num_frames=10)
    test_variance10 = [validation_labels[test_index][0], val_variances10]
    variance10_distances = []
    """
    test_variance20 = variance_list20[test_index[0]]
    variance20_distances = []
    test_variance30 = variance_list30[test_index[0]]
    variance30_distances = []
    test_variance40 = variance_list40[test_index[0]]
    variance40_distances = []
    test_variance50 = variance_list50[test_index[0]]
    variance50_distances = []
    test_variance100 = variance_list100[test_index[0]]
    variance100_distances = []

    test_variance10x10 = variance_list10x10[test_index[0]]
    variance10x10_distances = []
    test_variance20x20 = variance_list20x20[test_index[0]]
    variance20x20_distances = []
    test_variance30x30 = variance_list30x30[test_index[0]]
    variance30x30_distances = []
    test_variance40x40 = variance_list40x40[test_index[0]]
    variance40x40_distances = []
    test_variance50x50 = variance_list50x50[test_index[0]]
    variance50x50_distances = []
    test_variance70x70 = variance_list70x70[test_index[0]]
    variance70x70_distances = []
    test_variance100x100 = variance_list100x100[test_index[0]]
    variance100x100_distances = []
    test_variance100x200 = variance_list100x200[test_index[0]]
    variance100x200_distances = []
    """

    for j in range(len(labels)):
        variance1_distances.append([variance_list1[j][0], abs(variance_list1[j][1][0] - test_variance1[1][0])])
        variance10_distances.append([variance_list10[j][0], abs(variance_list10[j][1][0] - test_variance10[1][0])])
        """
        variance20_distances.append([variance_list20[j][0], abs(variance_list20[j][1][0] - test_variance20[1][0])])
        variance30_distances.append([variance_list30[j][0], abs(variance_list30[j][1][0] - test_variance30[1][0])])
        variance40_distances.append([variance_list40[j][0], abs(variance_list40[j][1][0] - test_variance40[1][0])])
        variance50_distances.append([variance_list50[j][0], abs(variance_list50[j][1][0] - test_variance50[1][0])])
        variance100_distances.append([variance_list100[j][0], abs(variance_list100[j][1][0] - test_variance100[1][0])])

        variance10x10_distances.append([variance_list10x10[j][0], abs(variance_list10x10[j][1][0] - test_variance10x10[1][0])])
        variance20x20_distances.append([variance_list20x20[j][0], abs(variance_list20x20[j][1][0] - test_variance20x20[1][0])])
        variance30x30_distances.append([variance_list30x30[j][0], abs(variance_list30x30[j][1][0] - test_variance30x30[1][0])])
        variance40x40_distances.append([variance_list40x40[j][0], abs(variance_list40x40[j][1][0] - test_variance40x40[1][0])])
        variance50x50_distances.append([variance_list50x50[j][0], abs(variance_list50x50[j][1][0] - test_variance50x50[1][0])])
        variance70x70_distances.append([variance_list70x70[j][0], abs(variance_list70x70[j][1][0] - test_variance70x70[1][0])])
        variance100x100_distances.append([variance_list100x100[j][0], abs(variance_list100x100[j][1][0] - test_variance100x100[1][0])])
        variance100x200_distances.append([variance_list100x200[j][0], abs(variance_list100x200[j][1][0] - test_variance100x200[1][0])])
        """

    # prediction for current test image
    if knncalc(k_knn, variance1_distances) == test_variance1[0]: correct_variance1_preds += 1
    #if knncalc(k_knn, variance10_distances) == test_variance10[0]: correct_variance10_preds += 1
    pred_variance10 = knncalc(k_knn, entropy_distances)
    if pred_variance10 == 'genuine' and test_variance10[0] == 'genuine' or pred_variance10 == 'synthethic' and \
            test_variance10[0] == 'spoofed':
        correct_variance10_preds += 1
    """
    if knncalc(k_knn, variance20_distances) == test_variance20[0]: correct_variance20_preds += 1
    if knncalc(k_knn, variance30_distances) == test_variance30[0]: correct_variance30_preds += 1
    if knncalc(k_knn, variance40_distances) == test_variance40[0]: correct_variance40_preds += 1
    if knncalc(k_knn, variance50_distances) == test_variance50[0]: correct_variance50_preds += 1
    if knncalc(k_knn, variance100_distances) == test_variance100[0]: correct_variance100_preds += 1

    if knncalc(k_knn, variance10x10_distances) == test_variance10x10[0]: correct_variance10x10_preds += 1
    if knncalc(k_knn, variance20x20_distances) == test_variance20x20[0]: correct_variance20x20_preds += 1
    if knncalc(k_knn, variance30x30_distances) == test_variance30x30[0]: correct_variance30x30_preds += 1
    if knncalc(k_knn, variance40x40_distances) == test_variance40x40[0]: correct_variance40x40_preds += 1
    if knncalc(k_knn, variance50x50_distances) == test_variance50x50[0]: correct_variance50x50_preds += 1
    if knncalc(k_knn, variance70x70_distances) == test_variance70x70[0]: correct_variance70x70_preds += 1
    if knncalc(k_knn, variance100x100_distances) == test_variance100x100[0]: correct_variance100x100_preds += 1
    if knncalc(k_knn, variance100x200_distances) == test_variance100x200[0]: correct_variance100x200_preds += 1
    """


    """
    # -----------------------------------------------------------------------
    # HISTOGRAM
    # -----------------------------------------------------------------------

    #earthmovers distance (runs verrrry slow est.: 600h)
    em_distance = []
    for j in train_index:
        em = em_dist(images[test_index], images[j])
        em_distance.append([labels[j], em])

    # intersection distance (runs fast)
    it_distance = []
    for j in train_index:
        it = intersection_test(histograms[test_index], histograms[j])
        it_distance.append([labels[j], it])

    # euclidian distance
    ed_distance = []
    for j in train_index:
        ed = euclidean_distance_test(histograms[test_index][0], histograms[j])
        ed_distance.append([labels[j], ed])

    # sum of manhattan distances
    smd_distance = []
    for j in train_index:
        smd = sum_manhattan_distance_test(histograms[test_index][0], histograms[j])
        smd_distance.append([labels[j], smd])

    # prediction for current test image
    k_knn = 3
    pred_em = knncalc(k_knn, em_distance)
    pred_it = knncalc(k_knn, it_distance)
    pred_ed = knncalc(k_knn, ed_distance)
    pred_smd = knncalc(k_knn, smd_distance)

    pred_hist_combined = []
    pred_hist_combined.append([pred_it, pred_ed, pred_smd])
    pred_hist_combined_value = most_common_value(pred_hist_combined)

    if pred_em == labels[test_index]:
        correct_em_preds += 1

    if pred_it == labels[test_index]:
        correct_it_preds += 1

    if pred_ed == labels[test_index]:
        correct_ed_preds += 1

    if pred_smd == labels[test_index]:
        correct_smd_preds += 1

    if pred_hist_combined_value == labels[test_index]:
        correct_hist_combined_preds += 1
    """


total = len(validation_labels)
print(f'Total number of samples: {str(total)}')

print(f'Accuracy entropy: {correct_entropy_preds / total * 100:.2f}%')

print(f'Classification of data_PLUS_genuine and data_PLUS_003 \nVariances with knn {k_knn}')
print(f'Accuracy global variance noch falsch!: {correct_variance1_preds / total * 100:.3f}%')
print(f'Accuracy variance with 10 patches: {correct_variance10_preds / total * 100:.3f}%')
"""
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
"""

"""
accuracy_em = correct_em_preds / total
print(f'Accuracy em: {accuracy_em * 100:.2f}%')

accuracy_it = correct_it_preds / total
print(f'Accuracy it: {accuracy_it * 100:.2f}%')

accuracy_ed = correct_ed_preds / total
print(f'Accuracy ed: {accuracy_ed * 100:.2f}%')

accuracy_smd = correct_smd_preds / total
print(f'Accuracy smd: {accuracy_smd * 100:.2f}%')

accuracy_hist_combined = correct_hist_combined_preds / total
print(f"Accuracy combined of hist: {accuracy_hist_combined*100:.2f}%")
"""
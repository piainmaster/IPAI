import cv2
import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from skimage.measure import shannon_entropy


def calculate_entropy(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    entropy = shannon_entropy(image)
    return entropy


def load_images(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            entropy = calculate_entropy(image_path)
            images.append(entropy)
            labels.append(folder_path)
    return np.array(images), np.array(labels)


def main():
    spoof_folder = "E:\Data_prepared\PLUS\spoofed"
    original_folder = "E:\Data_prepared\PLUS\genuine"

    spoof_images, spoof_labels = load_images(spoof_folder)
    original_images, original_labels = load_images(original_folder)

    images = np.concatenate((spoof_images, original_images), axis=0)
    labels = np.concatenate((spoof_labels, original_labels), axis=0)

    loo = LeaveOneOut()

    correct_predictions = 0

    for train_index, test_index in loo.split(images):
        X_train, X_test = images[train_index], images[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # KNN classifier
        k = 3  # You can adjust the number of neighbors as needed
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train.reshape(-1, 1), y_train)

        # Test image
        test_image_path = "E:\Data_prepared\IDIAP\genuine\mest.png"
        test_entropy = calculate_entropy(test_image_path)
        test_label = knn_classifier.predict([[test_entropy]])

        if test_label == y_test[0]:
            correct_predictions += 1

    accuracy = correct_predictions / len(images)
    print(f"Accuracy with LOOCV: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

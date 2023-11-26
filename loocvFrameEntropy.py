#
import cv2
import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from skimage.measure import shannon_entropy


def calculate_entropies(image_path, num_frames):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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


def load_images(folder_path, num_frames):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            entropies = calculate_entropies(image_path, num_frames)
            images.append(entropies)
            labels.append(folder_path)
    return np.array(images), np.array(labels)


def main():
    spoof_folder = "E:\Data_prepared\PLUS\spoofed"
    original_folder = "E:\Data_prepared\PLUS\genuine"

    num_frames = 60
    spoof_images, spoof_labels = load_images(spoof_folder, num_frames)
    original_images, original_labels = load_images(original_folder, num_frames)

    # Use np.vstack to stack arrays vertically
    spoof_images = np.vstack(spoof_images)
    original_images = np.vstack(original_images)

    images = np.vstack((spoof_images, original_images))
    labels = np.concatenate((spoof_labels, original_labels), axis=0)

    loo = LeaveOneOut()

    correct_predictions = 0

    for train_index, test_index in loo.split(images):
        X_train, X_test = images[train_index], images[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # KNN classifier
        k = 5  # You can adjust the number of neighbors as needed
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)

        # Test image
        test_image_path = "E:\Data_prepared\IDIAP\genuine\mest.png"
        test_entropies = calculate_entropies(test_image_path, num_frames)
        test_label = knn_classifier.predict([test_entropies])

        if test_label == y_test[0]:
            correct_predictions += 1

    accuracy = correct_predictions / len(images)
    print(f"Accuracy with LOOCV: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()




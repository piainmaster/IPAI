

import numpy as np
import cv2
from scipy.stats import entropy
from PIL import Image
import matplotlib.pyplot as plt











image_path = ('D:\sample\c2.jpg')
image = cv2.imread(image_path)
 # converting image to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#  Computing the histogram

hist, _ = np.histogram(gray_image.ravel(), bins=128, range=(0, 128))
prob_dist = hist / hist.sum()
image_entropy = entropy(prob_dist, base=2)
print(f"Image Entropy {image_entropy}")
plt.hist(hist, density=1, bins=128)
plt.show()









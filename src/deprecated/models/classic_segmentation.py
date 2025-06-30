import cv2
import numpy as np
import matplotlib.pyplot as plt

image       = cv2.imread('../../data/examples/maps/tower_map.jpg')
gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred     = cv2.GaussianBlur(gray, (5, 5), 0)
edges       = cv2.Canny(blurred, 50, 150)
kernel      = np.ones((5,5), np.uint8)
closed      = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image[..., ::-1])
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Segmentation Mask")
plt.axis("off")

plt.show()

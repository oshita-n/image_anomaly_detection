import cv2
import numpy as np

image = cv2.imread("bottle/test/broken_large/011.png")
image = cv2.resize(image, dsize=(256, 256))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread("predict_images/predict_11.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

diff = np.uint8(np.abs(image.astype(np.float32) - image2.astype(np.float32)))
heatmap = cv2.applyColorMap(diff , cv2.COLORMAP_JET)
cv2.imwrite("heatmap.png", heatmap)
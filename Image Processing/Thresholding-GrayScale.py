import numpy as np
import matplotlib.pyplot as plt

# Define a custom colormap that maps values from 0 to 1 to grayscale
gray_map = plt.cm.colors.LinearSegmentedColormap.from_list('my_gray', [(0, 'black'), (1, 'white')])


# Load image
img = plt.imread(r'C:\Users\soham\Documents\GitHub\ML-Journey\Image Processing\input-images\image.jpg')

# Convert to grayscale using weighted average of RGB channels
gray = np.dot(img[..., :3], [1.0, 1.0, 1.0])

# Normalize values to between 0 and 1
gray = gray / 255.0

plt.imsave(r'C:\Users\soham\Documents\GitHub\ML-Journey\Image Processing\output-images\Thresholding-GrayScale-Result.png', gray, cmap=gray_map)

# Map scalar values to colors using the 'gray_map' colormap, which assigns grayscale colors to scalar values between 0 and 1.
plt.imshow(gray, cmap=gray_map)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
img = plt.imread(r'C:\Users\soham\Documents\GitHub\ML-Journey\Image Processing\input-images\image.jpg')
gray_map = plt.cm.colors.LinearSegmentedColormap.from_list('my_gray', [(0, 'black'), (1, 'white')])
# Convert the image to grayscale
gray_img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# Flatten the image to create a 1D array of pixel values
pixel_values = gray_img.reshape(-1)

# Initialize the K-Means clustering algorithm
kmeans = KMeans(n_clusters=3)

# Fit the K-Means clustering algorithm to the pixel values
kmeans.fit(pixel_values.reshape(-1, 1))

# Predict the cluster assignments for each pixel
cluster_assignments = kmeans.predict(pixel_values.reshape(-1, 1))

# Reshape the cluster assignments back into a 2D array
clustered_image = 1 - cluster_assignments.reshape(gray_img.shape)

plt.imsave(r'C:\Users\soham\Documents\GitHub\ML-Journey\Image Processing\output-images\Image-Segmentation-KMeans-Result.png', clustered_image, cmap=gray_map)

# Visualize the segmented image
plt.imshow(clustered_image, cmap='gray')
plt.show()

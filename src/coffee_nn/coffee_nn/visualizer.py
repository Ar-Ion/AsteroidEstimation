from matplotlib import colors
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

class Visualizer:
    def __init__(self, image_size, num_dimensions):
        self._image_size = image_size
        self._num_dimensions = num_dimensions
        self._count = 0
    
    def plot(self, coords, features):
        img = np.zeros((self._image_size, self._image_size, 3))

        mapper = np.linspace(0, 1, self._num_dimensions)
        hue = np.dot(features, mapper)
        value = np.linalg.norm(features, axis=1)

        hsv = np.array((hue, np.ones_like(hue), value))
        
        indices = (self._image_size*coords).astype(dtype=np.int32)
        img[indices[:, 0], indices[:, 1]] = colors.hsv_to_rgb(hsv.T)
        
        image = Image.fromarray((img * 255).astype(np.uint8))
        image.save(f"/home/arion/AsteroidModelVisualization/{self._count}.png")
        self._count += 1
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\fibo\3rd year_1st semester\THEOS-2\Path Planning\mockup\map2.png")
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height))

path_color = (255, 255, 255)
color_tolerance = 15 

def create_grid_map(image):
    grid = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            b, g, r = image[y, x]
            if all(abs(c1 - c2) <= color_tolerance for c1, c2 in zip((r, g, b), path_color)):
                grid[y, x] = 1  # Free space
    return grid

if __name__ == "__main__":
    grid = create_grid_map(image)
    max_y, max_x, _ = image.shape
    plt.imshow(grid, cmap="gray")
    plt.show()
    np.save("map2.npy", grid)
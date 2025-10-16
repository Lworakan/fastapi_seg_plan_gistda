import cv2

# Load image
image = cv2.imread(r"C:\fibo\3rd year_1st semester\THEOS-2\Path Planning\mockup\map2.png")

# Resize image to fit the screen (adjust scale factor as needed)
scale_percent = 50  # Resize to 50% of the original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height))

# Dictionary to store clicked points
clicked_points = {}

# Index trackers
n_index, s_index = 1, 1

# Mouse callback function
def click_event(event, x, y, flags, param):
    global n_index, s_index

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        if n_index <= 20:
            key = f"N{n_index}"
            n_index += 1
        elif s_index <= 15:
            key = f"S{s_index}"
            s_index += 1
        else:
            print("Maximum points reached.")
            return

        # Scale coordinates back to original image size
        original_x = int(x * 100 / scale_percent)
        original_y = int(y * 100 / scale_percent)
        
        clicked_points[key] = (original_x, original_y)
        print(f"{key}: {clicked_points[key]}")

        # Draw point on resized image
        cv2.circle(resized_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", resized_image)

# Display resized image
cv2.imshow("Image", resized_image)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Print final dictionary
print("Clicked points:", clicked_points)

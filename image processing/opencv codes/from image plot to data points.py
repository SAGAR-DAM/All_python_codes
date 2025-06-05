import cv2
import matplotlib.pyplot as plt
import numpy as np

# === STEP 1: Calibration ===
pixel_points = [(690,5), (8, 690)]   # (Top-right, Bottom-left)
data_points = [(-2, 0), (2, 0.04)]           # (x0, y0), (x1, y1)

def pixel_to_data(xp, yp, pixel_pts, data_pts,X,Y):
    (a, b), (c, d) = pixel_pts     # pixel points: (col, row)
    (x1, y1), (x2, y2) = data_pts      # data points: (x, y)

    A = (x1-x2)/(b-d)
    B = x1-b*A
    
    C = (y1-y2)/(a-c)
    D = y1-C*a
    
    x_data = A*yp+B
    y_data = C*xp+D
    

    return x_data, y_data



# === STEP 2: Load and Process Image ===
img = cv2.imread(r"C:\Users\mrsag\OneDrive\Desktop\desktop files\test plot for retrival.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === STEP 3: Extract and Convert Points ===
data_points_extracted = []
X,Y = img.shape[0], img.shape[1]
for cnt in contours:
    for point in cnt:
        x, y = point[0]
        data_x, data_y = pixel_to_data(y, x, pixel_points, data_points,X,Y)
        data_points_extracted.append((data_x, data_y))
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

# === STEP 4: Visualize Results ===
plt.figure(figsize=(10, 5))

# Show image with detected points
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Contours on Image")

# Plot extracted data
plt.subplot(1, 2, 2)
if data_points_extracted:
    x_data, y_data = zip(*data_points_extracted)
    plt.plot(x_data, y_data, 'b-', markersize=2)
    # plt.xscale("log")
    plt.xlabel('X axis (data)')
    plt.ylabel('Y axis (data)')
    plt.title("Extracted Data Points")
    plt.grid(True)
else:
    plt.title("No data points extracted")

plt.tight_layout()
plt.show()

# === STEP 5: Print Some Sample Points ===
print("Sample extracted data points:")
for i, pt in enumerate(data_points_extracted[:10]):
    print(f"{i+1}: x = {pt[0]:.2f}, y = {pt[1]:.2f}")

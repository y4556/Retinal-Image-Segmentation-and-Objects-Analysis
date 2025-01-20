import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd

def veins_mask(g_img):
    kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    dilated_img = cv.dilate(g_img, kernel_dilate)
    sub = g_img - dilated_img
    median_blur_img = cv.medianBlur(sub, 5)
    adaptive_thresh = cv.adaptiveThreshold(median_blur_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    median_blur_thresh = cv.medianBlur(adaptive_thresh, 5)
    not_img = cv.bitwise_not(median_blur_thresh)
    return not_img

# Finding out the Centre

def sum_filter(img1,m):
    r,c=img1.shape
    filtered_img=np.zeros_like(img1)
    h = m // 2
    w = m // 2
    padded_img = np.pad(img1, ((h, h), (w, w)), mode='constant')
    for i in range(r):
        for j in range(c):
            roi = padded_img[i:i + m, j:j + m]
            sum_roi = np.sum(roi)#taking the sum of the roi
            filtered_img[i, j] = sum_roi
    return filtered_img

def max_coordinate(filt_img):
    r,c=filt_img.shape
    print(r,c)
    max_pix=filt_img.max()#gives maximum filter
    print(max_pix)
    for i in range(r):
        for j in range(c):
            if filt_img[i,j]==max_pix:
                return i,j

def erode_mask(e_img):
    eroded = cv.erode(e_img, np.ones((5, 5)))
    _, binary_image = cv.threshold(eroded, 127, 1, cv.THRESH_BINARY)
    return binary_image


# Brightest Region
def extract_brightest_pixel(image, window_size):
    max_intensity = 0
    brightest_pixel = None
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            top_left = (max(0, x - window_size // 2), max(0, y - window_size // 2))
            bottom_right = (min(image.shape[1], x + window_size // 2), min(image.shape[0], y + window_size // 2))
            roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            intensity = roi.max()
            if intensity > max_intensity:
                max_intensity = intensity
                brightest_pixel = (x, y)
    # Adjust position to the center of the brightest region
    top_left = (max(0, brightest_pixel[0] - window_size // 2), max(0, brightest_pixel[1] - window_size // 2))
    bottom_right = (min(image.shape[1], brightest_pixel[0] + window_size // 2), min(image.shape[0], brightest_pixel[1] + window_size // 2))
    brightest_pixel = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)

    return brightest_pixel

def error_1(pixel_centre,x2,y2):
        x1, y1 = pixel_centre
        error = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        print("Diif",error)
        if error>100:
            return x1,y1
        else:
            return pixel_centre

def Error_Calculation(x2,y2,x1,y1):
    Error = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print("The error", Error)
    return Error




input_dir = "Assignment-2-20240428T163732Z-001/Fundus image"
output_dir = "output_images/"
coordinates_file = "Assignment-2-20240428T163732Z-001/optic_disc_centres.csv"
df = pd.read_csv(coordinates_file)
coordinates = {row['image']: (row['x'], row['y']) for _, row in df.iterrows()}
os.makedirs(output_dir, exist_ok=True)
image_files = os.listdir(input_dir)
errors_output_file = "errors.csv"
errors = []

for filename in image_files:
    if filename in coordinates:
     img_path = os.path.join(input_dir, filename)
     img = cv.imread(img_path)
     g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

     #Veins Maps
     veins_map = veins_mask(g_img)
     b_img = erode_mask(veins_map)
     img2 = sum_filter(b_img, 18)
     y2, x2 = max_coordinate(img2)

     #Brightest Region/ Pixel
     brightest = extract_brightest_pixel(g_img, 10)

     # Original Centre
     x1, y1 = coordinates[filename]

     #Error
     x_2, y_2 = error_1(brightest, x2, y2)
     error=Error_Calculation(x_2, y_2, x1, y1)
     errors.append({'Image': filename, 'Error': error})

     #Draw Circle and plot
     cv.circle(img, (brightest[0], brightest[1]), 15, (0, 0, 255), -1)  # Red dot with radius 15
     plt.plot(x2, y2, marker='x', color='red', markersize=10)
     plt.imshow(veins_map)
     plt.axis('on')

     #save output
     output_path = os.path.join(output_dir, filename)
     cv.imwrite(output_path, img)
     veins_map_output_path = os.path.join(output_dir, "veins_" + filename)
     cv.imwrite(veins_map_output_path, veins_map)
     cv.imshow("Original Image", img)
     cv.imshow("Resulting Mask", veins_map)
     plt.show()
     cv.waitKey(0)

errors_df = pd.DataFrame(errors)
errors_df.to_csv(errors_output_file, index=False)




# brightest=extract_brightest_pixel(g_img,10)
# print("Brightest pixel coordinate from the centre:",brightest)
# print("Center  coordinate from the Vein:",x2,y2)
# x1=486
# y1=268
# x_2,y_2=error_1(brightest,x2,y2)
# Error_Calculation(x_2,y_2,x1,y1)
# cv.imshow("Original Image",img)
# cv.imshow("Resulting Mask",veins_map)
# cv.circle(img, (brightest[0], brightest[1]), 15, (255, 0, 0), -1) # Red dot with radius 5
# cv.imshow("Image with Brightest Point", img)
# plt.plot(x2, y2, marker='x', color='red', markersize=10)
# plt.imshow(veins_map)
# plt.axis('on')
# plt.show()
# cv.waitKey()


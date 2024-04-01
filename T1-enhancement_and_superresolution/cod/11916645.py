# Name: Lucas Machado Marinho - 11916645
# Course code: SCC0251
# Year/Semester: 2024/ 1º Semester
# Title of the assignment: enhancement and superresolution

# ** Required imports **

import numpy as np
import imageio.v3 as imageio

# ** Function Definitions **

def superResolution(l1, l2, l3, l4):
    # Creating a zero matrix twice the size
    width, height = l1.shape
    H = np.zeros((width*2, height*2), dtype=l1.dtype)

    # Loop through each pixel in the original images and assign them to the corresponding positions in the high-resolution image
    # The comments on the lines of each insertion in H are due to data overrides in the logic.
    for i in range(width):
        for j in range(height):
            H[2*i, 2*j] = l1[i, j]            # Top-left corner
            H[2*i, 2*j + 1] = l2[i, j]        # Top-right corner
            H[2*i + 1, 2*j] = l3[i, j]        # Bottom-left corner
            H[2*i + 1, 2*j + 1] = l4[i, j]    # Bottom-right corner

    return H

def RMSE(H, Hsr):
    # Get the dimensions of the images
    N, M = H.shape
    
    # Calculate the sum of squared differences between corresponding pixels in the images
    sum_of_squares = np.sum((H.astype(float) - Hsr.astype(float)) ** 2)
    
    # Calculate the mean of the squared differences
    mean = sum_of_squares / (N * M)
    
    # Calculate the square root of the mean to obtain the Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean)
    
    return rmse

def histogram(A, no_levels):
    # gets the size of the input matrix
    N, M = A.shape
    # creates an empty histogram with size proportional to the number of graylevels 
    hist = np.zeros(no_levels).astype(int)

    # computes for all levels in the range
    for i in range(no_levels):
        # sum all positions in which A == i is true
        pixels_value_i = np.sum(A == i)
        # store it in the histogram array
        hist[i] = pixels_value_i
            
    return(hist)

def histogram_equalization(A, no_levels, hist=None):
    # If a histogram is not provided, calculate it
    if hist is None:
        hist, _ = np.histogram(A.flatten(), bins=no_levels, range=(0, no_levels-1))
    
    # creates an empty cumulative histogram
    histC = np.zeros(no_levels).astype(float)
    # compute the cumulative histogram
    histC[0] = hist[0] # first value (intensity 0)
    for i in range(1, no_levels):
        histC[i] = hist[i] + histC[i-1]

    # array to store the actual transformation function
    hist_transform = np.zeros(no_levels).astype(np.uint8)
    
    # input image size
    N, M = A.shape
    
    # create the image to store the equalized version
    A_eq = np.zeros([N, M]).astype(np.uint8)
    
    # for each intensity value, transform it into a new intensity
    for z in range(no_levels):
        s = (no_levels-1) * histC[z] / float(M*N)
        A_eq[A == z] = s
        hist_transform[z] = s
    
    return A_eq

def gamma_correction(image, gamma):
    # Normalize the image to values between 0 and 1
    normalized_image = image / 255.0
    
    # Apply gamma correction
    corrected_image = np.power(normalized_image, 1.0 / float(gamma))
    
    # Adjust the image back to the range of 0 to 255
    corrected_image = (corrected_image * 255).astype(np.uint8)
    
    return corrected_image

# ** Main Instructions **

base_name = input().rstrip()  # Input base name of the images
file_name = input().rstrip()  # Input file name of the reference image
enhancement_met_id = input().rstrip()  # Input enhancement method identifier
enhancement_met_param = input().rstrip()  # Input enhancement method parameter

# Generate file paths for the low-resolution images
img1 = base_name + str(0) + ".png"
img2 = base_name + str(1) + ".png"
img3 = base_name + str(2) + ".png"
img4 = base_name + str(0) + ".png"

# Load the low-resolution images and the reference high-resolution image
l1 = imageio.imread(img1)
l2 = imageio.imread(img2)
l3 = imageio.imread(img3)
l4 = imageio.imread(img4)
H = imageio.imread(file_name)

# Match the enhancement method identifier to perform the corresponding enhancement
match enhancement_met_id:
    # Perform super-resolution without any pre-processing
    case "0":
        Hsr = superResolution(l1, l2, l3, l4)
    
    # Perform the single-image cumulative histogram method
    case "1":
        l1 = histogram_equalization(l1, 256)
        l2 = histogram_equalization(l2, 256)
        l3 = histogram_equalization(l3, 256)
        l4 = histogram_equalization(l4, 256)
        Hsr = superResolution(l1, l2, l3, l4)

    # Perform the Joint Cumulative Histogram method
    case "2":
        histl1 = histogram(l1, 256)
        histl2 = histogram(l2, 256)
        histl3 = histogram(l3, 256)
        histl4 = histogram(l4, 256)
        meanHist = (histl1+histl2+histl3+histl4)/4
        
        l1 = histogram_equalization(l1, 256, meanHist)
        l2 = histogram_equalization(l2, 256, meanHist)
        l3 = histogram_equalization(l3, 256, meanHist)
        l4 = histogram_equalization(l4, 256, meanHist)
        Hsr = superResolution(l1, l2, l3, l4)
    
    # Perform gamma correction on low-resolution images before super-resolution
    case "3":
        l1 = gamma_correction(l1, enhancement_met_param)
        l2 = gamma_correction(l2, enhancement_met_param)
        l3 = gamma_correction(l3, enhancement_met_param)
        l4 = gamma_correction(l4, enhancement_met_param)
        Hsr = superResolution(l1, l2, l3, l4)

# Calculate and return the RMSE between the reference high-resolution image and the enhanced high-resolution image
RMSE_FORM = "{:.4f}".format(RMSE(H, Hsr))

print(RMSE_FORM)
## Depth-Anything model
## Eline van der Hoek
## 14-11-2024

## Input: train_x and train_y
## Output: dataframe containing all metrics, one dataframe will be saved for the ground truth results and one for the prediction results
    # Also a plot showing the distribution of the dice scores will be made and saved

import cv2
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn.functional as F
import seaborn as sns
from torchvision.transforms import Compose
from sklearn.cluster import KMeans
from matplotlib import colors
from pandas import DataFrame
from skimage import io, feature
from skimage.feature import graycomatrix
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
warnings.filterwarnings('ignore', category=DeprecationWarning)


# https://learnopencv.com/depth-anything/

def fill_oval_contours(mask, result):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = feature.canny(mask_gray, sigma=0)
    binary_image_edges = (edges * 255).astype(np.uint8)
    binary_image_edges = cv2.dilate(binary_image_edges, np.ones((1, 1), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(binary_image_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    height, width = mask_gray.shape

    def touches_border(contour, width, height):
        for point in contour:
            if point[0][0] <= 0 or point[0][0] >= width - 1 or point[0][1] <= 0 or point[0][1] >= height - 1:
                return True
        return False

    def fill_biggest_non_border_contour(cnts, width, height):
        filled_contours = np.zeros((height, width), dtype=np.uint8)
        kernel_3x3 = np.ones((3, 3), np.uint8)
        
        for contour in cnts:
            if len(contour) >= 3:
                if not touches_border(contour, width, height):
                    cv2.drawContours(filled_contours, [contour], -1, 255, cv2.FILLED)
                    
                    # Check for 5x5 matrix of values '1'
                    erosion = cv2.erode(filled_contours, kernel_3x3, iterations=1)
                    if np.any(erosion == 255):
                        return filled_contours
                    else:
                        filled_contours = np.zeros((height, width), dtype=np.uint8)
        return filled_contours

    filled_contours = fill_biggest_non_border_contour(cnts, width, height)
    
    # Refine the filled contour with erosion and dilation
    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(filled_contours, kernel, iterations=2)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)
    
    # Find contours again on the dilated image
    contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None

    # Create an expanded version of the filled contour to match the shape of 'result'
    contour_expanded_resized = np.expand_dims(img_dilation, axis=-1)  # Shape becomes (512, 512, 1)
    
    # Broadcast to match the shape of 'result'
    contour_expanded_resized = np.repeat(contour_expanded_resized, 3, axis=-1)  # Shape becomes (512, 512, 3)

    return contour, img_dilation, filled_contours

def find_k_optimal(image):
    image_8bit = (image * 255).astype(np.uint8)
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2HSV)

    h_channel = hsv_image[:, :, 0]
    
    # Step 3: Calculate the co-occurrence matrix T for H values
    cooc_matrix = graycomatrix(h_channel, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Step 4: Pick the diagonal values of co-occurrence matrix and store them in d
    diagonal_values = cooc_matrix[np.arange(256), np.arange(256), 0, 0]

    # Step 5: Find the local maximum from the diagonal values
    # i) finding the mean of the diagonal values of co-occurrence matrix
    mean_val = np.mean(diagonal_values)
    k_values = []

    k_min = 5
    k_max = 20

    for i in range(k_min, k_max + 1):
        K = np.sum(diagonal_values >= mean_val / i)
        k_values.append((K, i))
    
    optimal_k = max(k_values, key=lambda x: x[0])[1]
    
    #print(f'Number of optimal k:', optimal_k)

    # # Reshape the image to a 2D array of pixels
    # pixels = hsv_image.reshape((-1, 3))
    
    # # Define the range of k values to try
    # k_values = range(5, 15)
    
    # # Calculate inertia for each k value
    # inertias = []
    # for k in k_values:
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(pixels)
    #     inertias.append(kmeans.inertia_)
    
    # # Find the optimal k using the elbow method
    # optimal_k = 5  # Default value
    # for i in range(1, len(inertias) - 1):
    #     if (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1]) < 0.5:
    #         optimal_k = k_values[i]
    #         break

    # print(f'Number of optimal k:', optimal_k)
    return optimal_k

def thresholding_algo_kmeans(n_clusters, depth_color, image):
    img = np.array(depth_color, dtype=np.float64) / 255
    w, h, d = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))
    pixels = DataFrame(image_array, columns=["Red", "Green", "Blue"])
    pixels["colour"] = [colors.to_hex(p) for p in image_array]
    pixels_sample = pixels.sample(frac=0.25)

    kmeans = KMeans(n_clusters).fit(pixels_sample[["Red", "Green", "Blue"]])

    labels = kmeans.predict(pixels[["Red", "Green", "Blue"]])
    reduced_colors = np.array([kmeans.cluster_centers_[p] for p in labels])
    reduced_colors = np.clip(reduced_colors, 0, 1)  # Clip values to [0, 1] range
    reduced = np.reshape(reduced_colors, (w, h, d))

    reduced = cv2.convertScaleAbs(reduced, alpha=(255.0))

    mask_image = reduced
    mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
    #image = image*255
    image = image.astype(np.uint8)

    # Apply the mask to the original image
    overlaid_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

    # fig, axs = plt.subplots(1,3)
    # axs[0].imshow(image)
    # axs[1].imshow(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
    # axs[2].imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)) 
    # plt.show()   

    return reduced, mask_image

def find_contour_with_optimal_clusters(depth_color, image, initial_n_clusters, max_attempts=5):
    n_clusters = initial_n_clusters
    for attempt in range(max_attempts):
        reduced, mask_image = thresholding_algo_kmeans(n_clusters, depth_color, image)
        contour, img_dilation, filled_contour = fill_oval_contours(mask_image, image)
        
        if contour is not None:
            #print(f"Contour found with {n_clusters} clusters on attempt {attempt + 1}")
            return contour, img_dilation, filled_contour, n_clusters
        
        #print(f"No contour found with {n_clusters} clusters. Increasing n_clusters.")
        n_clusters += 1
    
    print(f"Failed to find contour after {max_attempts} attempts.")
    return None, None, None, n_clusters

import math
import matplotlib.pyplot as plt
from functions import load_data
from functions import preprocess_images
from functions import compute_dice_scores
from functions import calc_diam_area
from functions import visualize_outcome
from functions import extrapolate
from functions import depth_anything_function

if __name__ == "__main__":
    ## Uncomment if you want to train the algorithm
    # loc_img = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_x"
    # loc_mask = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_y"

    ## Uncomment if you want to test the algorithm
    # loc_img = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\test_x"
    # loc_mask = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\test_y"

    loc_output = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Python\\Depth-Anything-V2\\Output_Depth_anything\\Test"

    # DataFrame to store results
    dataframe = pd.DataFrame(columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])
    results = []
    results_gt = []
    finals = []
    dice_scores = []

    # Process each file in the input folder
    images, masks, filenames, images_notscaled, masks_notscaled = load_data(loc_img, loc_mask, target_size=(512,512))
    
    # Preprocess images 
    #IMPORTANT: uncomment 'extrapolate' within 'preprocess_images' function before use
    preprocessed_images, preprocessed_masks = preprocess_images(images, masks, target_size=(512,512))
    
    # Add channel dimension to masks
    preprocessed_masks = np.expand_dims(preprocessed_masks, axis=-1)

    num=0
    for i, (image, mask, f) in enumerate(zip(preprocessed_images, preprocessed_masks, filenames)):
        image = (image*255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        

        depth_gray, depth_color, final_result = depth_anything_function(image)

        # Thresholding with kmeans
        n_clusters = find_k_optimal(depth_color)
        contour, img_dilation, filled_contour, used_n_clusters = find_contour_with_optimal_clusters(depth_color, image, n_clusters)

        if contour is not None:
            pred_mask = cv2.resize(filled_contour, (512,512))
            
            dice_score = compute_dice_scores(np.expand_dims(mask, axis=0), np.expand_dims(pred_mask, axis=0))
            print(f"Dice score for image {f}: {dice_score[0]}")
            
            # for score in dice_score:
            if dice_score > 0.1:
                dice_scores.append(dice_score)
            
            # Calculate and store metrics
            print(f"Processing {i}: {f}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dA, dB, area, perimeter, final = calc_diam_area(pred_mask, image, f, loc_output)
            ratio = dA / dB if dB != 0 else float('inf')  # Prevent division by zero
            roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else float('inf')  # Prevent division by zero
            results.append([f, dA, dB, area, perimeter, ratio, roundness])
            finals.append(final)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dA, dB, area, perimeter, final = calc_diam_area(mask, image, f, loc_output)
            ratio = dA / dB if dB != 0 else float('inf')  # Prevent division by zero
            roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else float('inf')  # Prevent division by zero
            results_gt.append([f, dA, dB, area, perimeter, ratio, roundness])

            # Visualize the outcome
            mask = mask.squeeze()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            visualize_outcome(image, mask, pred_mask, loc_output, f)
        else:
            print(f"Failed to find contour for image {f}. Skipping...")
            results.append([f, None, None, None, None, None, None])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dA, dB, area, perimeter, final = calc_diam_area(mask, image, f, loc_output)
            ratio = dA / dB if dB != 0 else float('inf')  # Prevent division by zero
            roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else float('inf')  # Prevent division by zero
            results_gt.append([f, dA, dB, area, perimeter, ratio, roundness])
            num+=1

    # Convert results to DataFrame and print
    df = pd.DataFrame(results, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])
    df_gt = pd.DataFrame(results_gt, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])
    df.to_csv(f'{loc_output}/df_prediction.csv', index=False)  
    df_gt.to_csv(f'{loc_output}/df_ground_truth.csv', index=False)   

    # Print Dice scores
    dice_scores_array = np.array(dice_scores)
    print(len(dice_scores))
    print(np.mean(dice_scores))
    print(num)

    plt.figure(figsize=(8, 6))
    sns.histplot(dice_scores_array, kde=True, bins=20, color='blue')

    # Add labels and title
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dice Scores')

    # Show the plot
    plt.show()
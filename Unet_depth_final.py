## Unet using both depth-anything images as original images
## Eline van der Hoek
## 14-11-2024

## Input: train_x and train_y
## Output: dataframe containing all metrics, one dataframe will be saved for the ground truth results and one for the prediction results

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from functions import load_data, post_process_prediction, preprocess_images, calc_diam_area, calc_diam_area2, visualize_outcome, depth_anything_function
from functions import train_final_model_depth, cross_validation_depth

if __name__ == "__main__":
    ### Before running, change:
    # 1) extrapolate must be commented within the preprocess image function in the functions file

    ## Uncomment if you want to train the algorithm
    # loc_img = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_x"
    # loc_mask = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_y"

    ## Uncomment if you want to test the algorithm
    # loc_img = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\test_x"
    # loc_mask = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\test_y"

    loc_output = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Python\\Depth-Anything-V2\\Output_unet_depth_combi"

    # Load data
    original_images, masks, filenames, images_notscaled, masks_notscaled = load_data(loc_img, loc_mask, target_size=(128,128))
    
    # Preprocess images 
    preprocessed_images, preprocessed_masks = preprocess_images(original_images, masks, target_size=(128,128))
    
    # Add channel dimension to masks
    preprocessed_masks = np.expand_dims(preprocessed_masks, axis=-1)

    # Generate depth images
    depth_images = []
    for image in preprocessed_images:
        image = image.astype(np.float32)
        _, depth_image, _ = depth_anything_function(image)
        depth_images.append(depth_image)
    depth_images = np.array(depth_images)

    ## Uncomment this if you want to use cross-validation to train the model:
    #avg_dice_score, pred_masks, pred_masks_all_folds = cross_validation_depth(original_images, preprocessed_images, depth_images, preprocessed_masks, filenames, loc_output)
    #print(f"Average dice score across all folds: {avg_dice_score}")

    ## Uncomment this if you want to train final model
    #final_model, pred_masks, mean_dice = train_final_model_depth(original_images, preprocessed_images, depth_images, preprocessed_masks, filenames, loc_output)

    # Calculate metrics for ground truth
    results_gt = []
    finals_gt = []
    for mask, image, f in zip(masks, original_images, filenames):
        dA, dB, area, perimeter, final = calc_diam_area2(mask, image, f, loc_output)
        ratio = dA / dB if dB != 0 else float('inf')
        roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else float('inf')
        results_gt.append([f, dA, dB, area, perimeter, ratio, roundness])
        finals_gt.append(final)

    # Calculate metrics for predictions
    results_pred = []
    finals_pred = []
    for pred_mask, image, f in zip(pred_masks, original_images, filenames):
        dA, dB, area, perimeter, final = calc_diam_area(pred_mask, image, f, loc_output)
        ratio = dA / dB if dB != 0 else float('inf')
        roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else float('inf')
        results_pred.append([f, dA, dB, area, perimeter, ratio, roundness])
        finals_pred.append(final)

    # Create DataFrames and save to CSV
    df_gt = pd.DataFrame(results_gt, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])
    df_pred = pd.DataFrame(results_pred, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])

    ## Uncomment either train or test to save the data
    # df_gt.to_csv(f'{loc_output}/df_ground_truth_train.csv', index=False)
    # df_pred.to_csv(f'{loc_output}/df_prediction_train.csv', index=False)

    # df_gt.to_csv(f'{loc_output}/df_ground_truth_test.csv', index=False)
    # df_pred.to_csv(f'{loc_output}/df_prediction_test.csv', index=False)

    print("Processing completed. Results saved in the output directory.")
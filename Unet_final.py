## Standard U-Net
## Eline van der Hoek
## 14-11-2024

## Input: train_x and train_y
## Output: dataframe containing all metrics, one dataframe will be saved for the ground truth results and one for the prediction results

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from functions import load_data
from functions import post_process_prediction
from functions import cross_validation
from functions import preprocess_images
from functions import calc_diam_area, calc_diam_area2
from functions import visualize_outcome
from functions import train_final_unet_model

if __name__ == "__main__":

    loc_img = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_x"
    loc_mask = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_y"
    loc_output = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Python\\Depth-Anything-V2\\Output_unet"

    # Load data
    images, masks, filenames, images_notscaled, masks_notscaled = load_data(loc_img, loc_mask, target_size=(128,128))
    
    # Preprocess images 
    preprocessed_images, preprocessed_masks = preprocess_images(images, masks, target_size=(128,128))
    
    # Add channel dimension to masks
    preprocessed_masks = np.expand_dims(preprocessed_masks, axis=-1)

    ## Uncomment this if you want to use cross-validation to train the model:
    #avg_dice_score, pred_masks, X_val, pred_masks_all_folds = cross_validation(original_images, images, masks, filenames, loc_output)
    #print(f"Average dice score across all folds: {avg_dice_score}")

    ## Uncomment this if you want to train final model
    #final_model, pred_masks, final_dice_score = train_final_unet_model(images_notscaled, preprocessed_images, preprocessed_masks, filenames, loc_output)
    #print(f"Final Dice Score: {final_dice_score}")

    # Calculate metrics for predicted masks
    results = []
    finals = []
    for i, (pred_mask, image, f) in enumerate(zip(pred_masks, images_notscaled, filenames)):
        dA, dB, area, perimeter, final = calc_diam_area(pred_mask, image, f, loc_output)
        ratio = dA/dB if dB != 0 else float('inf')  # Prevent division by zero
        roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else float('inf')  # Prevent division by zero
        results.append([f, dA, dB, area, perimeter, ratio, roundness])
        finals.append(final)

    # Calculate metrics for ground truth masks
    results_gt = []
    finals_gt = []
    for i, (mask, image, f) in enumerate(zip(masks, images_notscaled, filenames)):
        dA_gt, dB_gt, area_gt, perimeter_gt, final_gt = calc_diam_area2(mask, image, f, loc_output)
        ratio_gt = dA_gt/dB_gt if dB_gt != 0 else float('inf')  # Prevent division by zero
        roundness_gt = 4 * math.pi * area_gt / (perimeter_gt ** 2) if perimeter_gt != 0 else float('inf')  # Prevent division by zero
        results_gt.append([f, dA_gt, dB_gt, area_gt, perimeter_gt, ratio_gt, roundness_gt])
        finals_gt.append(final_gt)

    # Save results to CSV
    df_pred = pd.DataFrame(results, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])
    df_gt = pd.DataFrame(results_gt, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])

    df_pred.to_csv(f'{loc_output}/df_prediction.csv', index=False)  
    df_gt.to_csv(f'{loc_output}/df_ground_truth.csv', index=False)

    ## Uncomment either train or test to save the data    
    # df_gt.to_csv(f'{loc_output}/df_ground_truth_train.csv', index=False)
    # df_pred.to_csv(f'{loc_output}/df_prediction_train.csv', index=False)
    
    # df_gt.to_csv(f'{loc_output}/df_ground_truth_test.csv', index=False)
    # df_pred.to_csv(f'{loc_output}/df_prediction_test.csv', index=False)

    print("Processing completed. Results saved in the output directory.")

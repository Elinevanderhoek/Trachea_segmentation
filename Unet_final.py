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
    # Within the preprocess_images function:
    #     - anisotropic blurring should be UNcommented
    #     - extrapolation and histogram equalization should be commented
    
    loc_img = "...\\train_x"
    loc_mask = "...\\train_y"
    loc_output = "...\\Output_unet"

    # Load data
    images, masks, filenames, images_notscaled, masks_notscaled = load_data(loc_img, loc_mask, target_size=(128,128))
    
    # Preprocess images 
    preprocessed_images, preprocessed_masks = preprocess_images(images, masks, target_size=(128,128))
    
    # Add channel dimension to masks
    preprocessed_masks = np.expand_dims(preprocessed_masks, axis=-1)

    # Train the final U-Net model
    final_model, pred_masks, final_dice_score = train_final_unet_model(images_notscaled, preprocessed_images, preprocessed_masks, filenames, loc_output)

    print(f"Final Dice Score: {final_dice_score}")

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
    results2 = []
    finals2 = []
    for i, (mask, image, f) in enumerate(zip(masks, images_notscaled, filenames)):
        dA2, dB2, area2, perimeter2, final2 = calc_diam_area2(mask, image, f, loc_output)
        ratio2 = dA2/dB2 if dB2 != 0 else float('inf')  # Prevent division by zero
        roundness = 4 * math.pi * area2 / (perimeter2 ** 2) if perimeter2 != 0 else float('inf')  # Prevent division by zero
        results2.append([f, dA2, dB2, area2, perimeter2, ratio2, roundness])
        finals2.append(final2)

    # Save results to CSV
    df = pd.DataFrame(results, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])
    df2 = pd.DataFrame(results2, columns=['Image Name', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio', 'Roundness'])
    df.to_csv(f'{loc_output}/df_prediction.csv', index=False)  
    df2.to_csv(f'{loc_output}/df_ground_truth.csv', index=False)

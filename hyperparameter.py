## Find hyperparameters by using the Hyperband algorithm
## Eline van der Hoek
## 14-11-2024

## Input: train_x and train_y
    # It is important to change the preprocess_images function so that it uses the right functions (extrapolation, normalization, etc.)
## Output: 
    # The optimal number of filters in the first conv layer is ... 
    # The optimal number of filters in the 2nd conv layer is ...
    # The optimal number of filters in the 3rd conv layer is ...
    # The optimal number of bottleneck_filters is ...
    # The optimal kernel size is ...
    # The optimal learning rate is ...
    # The optimal batch size is ...
    # The optimal optimizer is ...
    # The optimal number of epochs is ...

    # Change the hyperparameter values in the functions file for the corresponding model
    
    # It also plots a loss and learning curve for the ground truth and validation data, with standard deviations displayed (shaded) 

import numpy as np
import cv2
import os
import skimage
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
import torch
import torch.nn.functional as F
matplotlib.use('TkAgg')  # Use a interactive backend
import keras_tuner as kt
from keras_tuner import Objective
from torchvision.transforms import Compose
import pandas as pd
import random
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from scipy.ndimage import binary_fill_holes
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from PIL import Image, ImageDraw
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
warnings.filterwarnings('ignore', category=DeprecationWarning)

def crop_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7),0)

    # Threshold the image to separate the circular region from the background
    _, binary = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)

    mask_array = np.array(binary)
    mask_image = binary_fill_holes(mask_array)
    mask_image = binary_erosion(mask_image, structure=np.ones((10,10)))
    mask_image = binary_dilation(mask_image, structure=np.ones((10,10)))
    mask_array = np.array(mask_image).astype(np.uint8)

    contours = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)

    img_cropped = image[y:y+h, x:x+w]
    mask_cropped = mask_image[y:y+h, x:x+w]

    return img_cropped, mask_cropped

def extrapolate(image, mask):
    # Ensure mask is binary
    struct1 = ndimage.generate_binary_structure(2,2)
    mask = binary_erosion(mask, structure = struct1, iterations=10)
    mask = np.invert(mask)

    def closest_point_extrapolation(image, mask):
        height, width = mask.shape
        output = image.copy()
        
        distances, indices = distance_transform_edt(mask, return_indices=True)
        
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    nearest_y, nearest_x = indices[:, y, x]
                    output[y, x] = image[nearest_y, nearest_x]
        
        return output

    extrapolated_closest = closest_point_extrapolation(image, mask)
    result_image = extrapolated_closest.copy()
    x,y,z = result_image.shape
    result_image = result_image[15:x-15, 15:x-10]

    # fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    # ax[0].imshow(image)
    # ax[0].set_title('Original Image')
    # ax[1].imshow(extrapolated_closest)
    # ax[1].set_title('Closest Point Extrapolation')
    # ax[2].imshow(mask, cmap='Greys')
    # ax[2].set_title('Mask')
    # plt.show()

    return result_image

def load_data(loc_img, loc_mask, target_size=(128,128)):
    images = []
    masks = []

    for file in os.listdir(loc_img):
        f = os.path.join(loc_img, file)
        image = cv2.imread(f, cv2.IMREAD_COLOR)
        image = cv2.resize(image, target_size)
        
        images.append(image)

    for file in os.listdir(loc_mask):
        f = os.path.join(loc_mask, file)
        mask = nib.load(f)
        mask = np.array(mask.dataobj)
        mask = mask.transpose(1, 0, 2)
        mask = cv2.resize(mask, image.shape[:2])
        mask = cv2.resize(mask, target_size)
        mask = (mask > 0.5).astype(np.float32)
        masks.append(mask)
    return np.array(images), np.array(masks)

def preprocess_images(images, target_size=(128,128)):
    preprocessed_images = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, mask1 = crop_image(image)
        # image = extrapolate(image, mask1)
        image = cv2.resize(image, target_size)
        image = np.array(image)/255

        # # Adaptive Equalization
        #image = exposure.equalize_adapthist(image, clip_limit=0.01)

        preprocessed_images.append(image)
        
    return np.array(preprocessed_images)

def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def touches_border(contour, width, height):
    return np.any(contour[:, 0, 0] == 0) or np.any(contour[:, 0, 0] == width - 1) or \
           np.any(contour[:, 0, 1] == 0) or np.any(contour[:, 0, 1] == height - 1)

def fill_biggest_non_border_contour(cnts, width, height, min_area=50):
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

def create_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation((-0.2,0.2)),
        layers.RandomZoom((0.05,0.2),None),
    ])

def augment_data(X_rgb, y, X_depth=None, augmentation_factor=5):
    augmentation_layer = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation((-0.2,0.2)),
        layers.RandomZoom((0.05,0.2),None),])
    augmented_rgb = []
    augmented_depth = [] if X_depth is not None else None
    augmented_y = []

    for _ in range(augmentation_factor):
            for i in range(len(X_rgb)):
                image = tf.convert_to_tensor(X_rgb[i:i+1], dtype=tf.float32)
                mask = tf.convert_to_tensor(y[i:i+1], dtype=tf.float32)
                
                # Combine RGB and mask
                combined_rgb = tf.concat([image, mask], axis=-1)

                # If depth data is provided, combine RGB, depth, and mask
                if X_depth is not None:
                    depth = tf.convert_to_tensor(X_depth[i:i+1], dtype=tf.float32)
                    combined = tf.concat([combined_rgb, depth], axis=-1)
                else:
                    combined = combined_rgb
                
                # Apply augmentation
                augmented_combined = augmentation_layer(combined, training=True)

                # Split the augmented data back into RGB, depth (if available), and mask
                if X_depth is not None:
                    num_channels_rgb = image.shape[-1]
                    num_channels_depth = depth.shape[-1]
                    augmented_rgb_image = augmented_combined[..., :num_channels_rgb]
                    augmented_depth_image = augmented_combined[..., num_channels_rgb:num_channels_rgb + num_channels_depth]
                    augmented_mask = augmented_combined[..., -1:]
                    augmented_rgb.append(augmented_rgb_image[0].numpy())
                    augmented_depth.append(augmented_depth_image[0].numpy())
                else:
                    num_channels_rgb = image.shape[-1]
                    augmented_rgb_image = augmented_combined[..., :num_channels_rgb]
                    augmented_mask = augmented_combined[..., num_channels_rgb:]
                    augmented_rgb.append(augmented_rgb_image[0].numpy())

                augmented_y.append(augmented_mask[0].numpy())

    if X_depth is not None:
        return np.array(augmented_rgb), np.array(augmented_depth), np.array(augmented_y)
    else:
        return np.array(augmented_rgb), np.array(augmented_y)

def visualize_results(images, true_masks, pred_masks, pred_masks_post_processing, num_examples=3):
    """
    Visualize the original image, ground truth mask, and predicted mask for a given number of examples.
    """
    fig, axes = plt.subplots(num_examples, 4, figsize=(10, 3*num_examples))
    
    for i in range(num_examples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        true_mask = np.squeeze(true_masks[i])
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred_mask = np.squeeze(pred_masks[i])
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

        # Predicted mask
        pred_mask_post_processing = np.squeeze(pred_masks_post_processing[i])
        axes[i, 3].imshow(pred_mask_post_processing, cmap='gray')
        axes[i, 3].set_title('Predicted Mask Post Processing')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history, fold):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(f'Loss - Fold {fold+1}')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Train Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
    plt.legend()
    plt.title(f'Dice Coefficient - Fold {fold+1}')
    plt.show()

def compute_dice_scores(y_val, pred_masks):
    dice_scores = []
    for true, pred in zip(y_val, pred_masks):
        score = dice_coefficient(tf.convert_to_tensor(true, dtype=tf.float32), tf.convert_to_tensor(pred, dtype=tf.float32)).numpy()
        dice_scores.append(score)
    return dice_scores

def post_process_prediction(pred_mask, kernel_size, iterations):
    threshold = threshold=np.mean(pred_mask)

    # Apply threshold
    binary_mask = (pred_mask.squeeze() > threshold).astype(np.uint8)

    # Morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=iterations)
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)

    return dilated

def cross_validation(images, masks, n_splits=5, batch_size=32, kernel_sizes=range(1, 11), iterations_range=range(1, 11)):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_fold_scores = []
    best_dice_scores = []
    best_dice_score = 0
    best_kernel_size = None
    best_iterations = None
    
    for kernel_size in kernel_sizes:
        for iterations in iterations_range:
            print(f"Testing with kernel_size={kernel_size}, iterations={iterations}")
            fold_scores = []
            dice_scores = []
            for fold, (train_index, val_index) in enumerate(kf.split(images)):
                print(f"Training on fold {fold+1}/{n_splits}")
                
                X_train, X_val = images[train_index], images[val_index]
                y_train, y_val = masks[train_index], masks[val_index]
                X_train, y_train = augment_data(X_train, y_train, augmentation_factor=5)

                print(f"X_val shape: {X_val.shape}")
                print(f"y_val shape: {y_val.shape}")

                model = unet_model()

                model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss="binary_crossentropy",
                            metrics=["accuracy", dice_coefficient])
                
                callbacks = [
                    ModelCheckpoint(f'best_model_fold_{fold+1}.keras', save_best_only=True, monitor='val_dice_coefficient', mode='max'),
                    EarlyStopping(patience=10, monitor='val_dice_coefficient', mode='max'),
                    ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.2, patience=5, min_lr=1e-6)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    callbacks=callbacks
                )
                
                # Evaluate on validation set
                val_loss, val_acc, val_dice = model.evaluate(X_val,y_val)
                fold_scores.append(val_dice)
                
                # Predict on validation set
                pred_masks = model.predict(X_val)
                pred_masks_post_processing = np.array([post_process_prediction(mask, kernel_size, iterations) for mask in pred_masks])
                dice_score = compute_dice_scores(y_val, pred_masks_post_processing)

                mean_dice = np.mean(dice_score)
                dice_scores.append(mean_dice)
                
                print(f"Fold {fold+1} - Validation Dice: {val_dice}")
                print(f"Fold {fold+1} - Post-processed Validation Dice: {mean_dice}")
                
                # Visualize results for 5 random examples
                random_indices = np.random.choice(len(X_val), 5, replace=False)
                visualize_results(X_val[random_indices], y_val[random_indices], pred_masks[random_indices], pred_masks_post_processing[random_indices])        
                
                # Plot learning curves for this fold
                plot_learning_curves(history, fold)

            avg_dice_score = np.mean(dice_scores)
            print(f"Kernel Size: {kernel_size}, Iterations: {iterations}, Average Dice Score: {avg_dice_score}")

            if avg_dice_score > best_dice_score:
                best_dice_score = avg_dice_score
                best_kernel_size = kernel_size
                best_iterations = iterations
                best_fold_scores = fold_scores
                best_dice_scores = dice_scores

    print(f"Best Dice Score: {best_dice_score} with kernel_size={best_kernel_size} and iterations={best_iterations}")
    return best_fold_scores, best_dice_scores

def unet_model(hp):
    # Define the hyperparameters to tune
    kernel_size = hp.Choice('kernel_size', values=[1, 3, 5,7])
    conv_filters_1 = hp.Int('conv1_filters', min_value=16, max_value=64, step=16)
    conv_filters_2 = hp.Int('conv2_filters', min_value=32, max_value=128, step=32)
    conv_filters_3 = hp.Int('conv3_filters', min_value=64, max_value=256, step=64)
    bottleneck_filters = hp.Int('bottleneck_filters', min_value=128, max_value=512, step=128)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    batch_size = hp.Choice('batch_size', values=[8, 12, 16, 20, 24, 28, 32])
    epochs = hp.Int('epochs', min_value=10, max_value=50, step=5) 

    inputs = Input(shape=(128, 128, 3))

    # Encoder
    c1 = Conv2D(conv_filters_1, kernel_size, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(conv_filters_2, kernel_size, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(conv_filters_3, kernel_size, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(bottleneck_filters, kernel_size, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate([u5, c3])
    c5 = Conv2D(conv_filters_3, kernel_size, padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate([u6, c2])
    c6 = Conv2D(conv_filters_2, kernel_size, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate([u7, c1])
    c7 = Conv2D(conv_filters_1, kernel_size, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs=inputs, outputs=[outputs])
    
    # Select optimizer based on the hyperparameter choice
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])
    return model

def unet_model_depth_input(hp):
    # Define the hyperparameters to tune
    kernel_size = hp.Choice('kernel_size', values=[1, 3, 5,7])
    conv_filters_1 = hp.Int('conv1_filters', min_value=16, max_value=64, step=16)
    conv_filters_2 = hp.Int('conv2_filters', min_value=32, max_value=128, step=32)
    conv_filters_3 = hp.Int('conv3_filters', min_value=64, max_value=256, step=64)
    bottleneck_filters = hp.Int('bottleneck_filters', min_value=128, max_value=512, step=128)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    batch_size = hp.Choice('batch_size', values=[8, 12, 16, 20, 24, 28, 32])
    epochs = hp.Int('epochs', min_value=10, max_value=50, step=5) 

    inputs = Input(shape=(128, 128, 3))

    # Encoder
    c1 = Conv2D(conv_filters_1, kernel_size, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(conv_filters_2, kernel_size, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(conv_filters_3, kernel_size, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(bottleneck_filters, kernel_size, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(conv_filters_3, kernel_size, padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(conv_filters_2, kernel_size, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(conv_filters_1, kernel_size, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs=inputs, outputs=[outputs])
    
    # Select optimizer based on the hyperparameter choice
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])
    return model

def unet_model_with_depth(hp):
    # Define the hyperparameters to tune
    kernel_size = hp.Choice('kernel_size', values=[1, 3, 5,7])
    conv_filters_1 = hp.Int('conv1_filters', min_value=16, max_value=64, step=16)
    conv_filters_2 = hp.Int('conv2_filters', min_value=32, max_value=128, step=32)
    conv_filters_3 = hp.Int('conv3_filters', min_value=64, max_value=256, step=64)
    bottleneck_filters = hp.Int('bottleneck_filters', min_value=128, max_value=512, step=128)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    batch_size = hp.Choice('batch_size', values=[8, 12, 16, 20, 24, 28, 32])
    epochs = hp.Int('epochs', min_value=10, max_value=50, step=5) 
    
    input_size = (128,128,3)
    inputs_rgb = Input(input_size, name='input_rgb')
    inputs_depth = Input(input_size, name='input_depth')
    
    # Encoder for RGB
    c1_rgb = Conv2D(conv_filters_1, kernel_size, padding='same')(inputs_rgb)
    c1_rgb = BatchNormalization()(c1_rgb)
    c1_rgb = Activation('relu')(c1_rgb)
    p1_rgb = MaxPooling2D((2, 2))(c1_rgb)
    
    c2_rgb = Conv2D(conv_filters_2, kernel_size, padding='same')(p1_rgb)
    c2_rgb = BatchNormalization()(c2_rgb)
    c2_rgb = Activation('relu')(c2_rgb)
    p2_rgb = MaxPooling2D((2, 2))(c2_rgb)
    
    c3_rgb = Conv2D(conv_filters_3, kernel_size, padding='same')(p2_rgb)
    c3_rgb = BatchNormalization()(c3_rgb)
    c3_rgb = Activation('relu')(c3_rgb)
    p3_rgb = MaxPooling2D((2, 2))(c3_rgb)
    
    # Encoder for Depth
    c1_depth = Conv2D(conv_filters_1, kernel_size, padding='same')(inputs_depth)
    c1_depth = BatchNormalization()(c1_depth)
    c1_depth = Activation('relu')(c1_depth)
    p1_depth = MaxPooling2D((2, 2))(c1_depth)
    
    c2_depth = Conv2D(conv_filters_2, kernel_size, padding='same')(p1_depth)
    c2_depth = BatchNormalization()(c2_depth)
    c2_depth = Activation('relu')(c2_depth)
    p2_depth = MaxPooling2D((2, 2))(c2_depth)
    
    # Add another layer for the Depth encoder
    c3_depth = Conv2D(conv_filters_3, kernel_size, padding='same')(p2_depth)
    c3_depth = BatchNormalization()(c3_depth)
    c3_depth = Activation('relu')(c3_depth)
    p3_depth = MaxPooling2D((2, 2))(c3_depth)
    
    # Merge Encoders
    merge_bridges = Concatenate()([p3_rgb, p3_depth])
    
    # Bottleneck
    c4 = Conv2D(bottleneck_filters, kernel_size, padding='same')(merge_bridges)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    merge5 = Concatenate()([u5, c3_rgb, c3_depth])
    c5 = Conv2D(conv_filters_3, kernel_size, padding='same')(merge5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    merge6 = Concatenate()([u6, c2_rgb, c2_depth])
    c6 = Conv2D(conv_filters_2, kernel_size, padding='same')(merge6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    merge7 = Concatenate()([u7, c1_rgb, c1_depth])
    c7 = Conv2D(conv_filters_1, kernel_size, padding='same')(merge7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)

    model = Model(inputs=[inputs_rgb, inputs_depth], outputs=[outputs])
    
    # Select optimizer based on the hyperparameter choice
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])
    
    return model

def depth_anything_function(result):
    img = result
    max_depth = 20
    dataset = 'hypersim'
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    #encoders = ['vits', 'vitb', 'vitl']
    encoder = 'vits'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth'))
    depth_anything.eval()

    margin_width = 50
    caption_height = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    raw_image = img.copy()
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    h, w = image.shape[:2]
        
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
    with torch.no_grad():
        depth = depth_anything(image)
        
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
    depth_gray = depth.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
    
    split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
    raw_image_uint8 = cv2.convertScaleAbs(raw_image, alpha=(255.0))
    combined_results = cv2.hconcat([raw_image_uint8, split_region, depth_color])

    caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
    captions = ['Raw image', 'Depth Anything']
    segment_width = w + margin_width
    for i, caption in enumerate(captions):
        # Calculate text size
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]
        # Calculate x-coordinate to center the text
        text_x = int((segment_width * i) + (w - text_size[0]) / 2)

        # Add text caption
        cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
        
    final_result = cv2.vconcat([caption_space, combined_results])
    x,y,z = result.shape
    depth_color = cv2.resize(depth_color, (y,x))
    depth_gray = cv2.resize(depth_gray,(y,x))

    # put mask into alpha channel of result
    result_color = depth_color.copy()

    result_gray = depth_gray.copy()

    return result_gray, result_color, final_result

loc_img = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_x"
loc_mask = "C:\\Users\\ehoek3\\OneDrive - UMC Utrecht\\Tracheomalacie\\Frames\\Alle_frames\\train_y"

# Load data
images, masks = load_data(loc_img, loc_mask)

# Preprocess images once
preprocessed_images = preprocess_images(images)

# Add channel dimension to masks
preprocessed_masks = np.expand_dims(masks, axis=-1)

depth_images= []
for image in images:
    result_gray, depth_image, overlay = depth_anything_function(image)  # function to create depth maps
    depth_images.append(depth_image)
np.array(depth_images)

## For standard u-net
    # X_train, X_val, y_train, y_val = train_test_split(preprocessed_images, preprocessed_masks, test_size=0.15, random_state=42)
    # X_train = np.array(X_train)
    # X_val = np.array(X_val)

## For u-net using depth-anything images
    # X_train, X_val, y_train, y_val = train_test_split(depth_images, preprocessed_masks, test_size=0.15, random_state=42)
    # X_train = np.array(X_train)
    # X_val = np.array(X_val)

## For u-net using both depth-anything and original images
    # X_train_rgb, X_val_rgb, X_train_depth, X_val_depth, y_train, y_val = train_test_split(
    #     preprocessed_images, depth_images, preprocessed_masks, test_size=0.15, random_state=42)

    # X_train_depth = np.array(X_train_depth)
    # X_train_rgb, y_train = augment_data(X_train_rgb, y_train, augmentation_factor=5)
    # X_train_depth, _ = augment_data(X_train_depth, y_train, augmentation_factor=5)  # Augment depth data similarly

    # # Combine RGB and Depth data for training and validation
    # X_train = [np.array(X_train_rgb), np.array(X_train_depth)]
    # X_val = [np.array(X_val_rgb), np.array(X_val_depth)]

objective = Objective("val_dice_coefficient", direction="max")

# Initialize the tuner
tuner = kt.Hyperband(
    unet_model_depth_input,
    objective=objective,        # Objective to optimize
    max_epochs=50,              # Maximum number of epochs
    factor=5,                   # Factor by which the number of epochs per trial is reduced
    overwrite=True,
)

tuner.search(X_train, y_train,
             epochs=50,
             validation_data=(X_val, y_val),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of filters in the first conv layer is {best_hps.get('conv1_filters')} 
The optimal number of filters in the 2nd conv layer is {best_hps.get('conv2_filters')} 
The optimal number of filters in the 3rd conv layer is {best_hps.get('conv3_filters')}
The optimal number of bottleneck_filters is {best_hps.get('bottleneck_filters')}
The optimal kernel size is {best_hps.get('kernel_size')} 
The optimal learning rate is {best_hps.get('learning_rate')} 
The optimal batch size is {best_hps.get('batch_size')} 
The optimal optimizer is {best_hps.get('optimizer')}
The optimal number of epochs is {best_hps.get('epochs')}
""")

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=20, 
                    validation_data=(X_val, y_val),
                    batch_size=best_hps.get('batch_size'))

# Initialize an empty list to collect trial data
trial_data = []

# Loop through all trials
for trial in tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials)):
    # Extract hyperparameter values
    hp_values = trial.hyperparameters.values.copy()
    # Extract the best value of the metric (e.g., val_dice_coefficient)
    best_metric = trial.metrics.get_best_value('val_dice_coefficient')
    
    # Combine into a single dictionary
    hp_values['val_dice_coefficient'] = best_metric
    
    # Append to the list
    trial_data.append(hp_values)

# Convert the list of dictionaries to a DataFrame
df_results = pd.DataFrame(trial_data)
df_results['trial_id'] = range(len(df_results))
print(df_results.columns)
print(df_results.head())

plt.figure(figsize=(10, 6))
sns.lineplot(x='trial_id', y='val_dice_coefficient', hue='batch_size', data=df_results)
plt.title('All Trials: Validation Dice Coefficient')
plt.xlabel('Trial ID')
plt.ylabel('Validation Dice Coefficient')
plt.show()


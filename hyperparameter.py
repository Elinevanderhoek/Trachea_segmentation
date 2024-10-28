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

loc_img = "...\\train_x"
loc_mask = "...\\train_y"

# Load data
images, masks = load_data(loc_img, loc_mask)

# Preprocess images once
preprocessed_images = preprocess_images(images)

# Add channel dimension to masks
preprocessed_masks = np.expand_dims(masks, axis=-1)

depth_images= []
for image in images:
    result_gray, depth_image, overlay = depth_anything_function(image)  # Your function to create depth maps
    depth_images.append(depth_image)
np.array(depth_images)

X_train, X_val, y_train, y_val = train_test_split(depth_images, preprocessed_masks, test_size=0.15, random_state=42)
X_train = np.array(X_train)
X_val = np.array(X_val)

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

def plot_hyperparameter_impact(df_results, hyperparameter_name):
    plt.figure(figsize=(10, 6))
    for value in df_results[hyperparameter_name].unique():
        subset = df_results[df_results[hyperparameter_name] == value]
        plt.plot(subset.index, subset['val_dice_coefficient'], label=f"{hyperparameter_name}={value}")

    plt.xlabel('Trial')
    plt.ylabel('Validation Dice Coefficient')
    plt.title(f'Impact of {hyperparameter_name} on Validation Dice Coefficient')
    plt.legend()
    plt.show()

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

# # Plot the impact of different hyperparameters
# for hp_name in ['kernel_size', 'conv1_filters', 'conv2_filters', 'conv3_filters', 'batch_size', 'learning_rate', 'epochs', 'optimizer']:
#     plot_hyperparameter_impact(df_results, hp_name)

numeric_cols = df_results.select_dtypes(include=[np.number]).columns
correlation_matrix = df_results[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Hyperparameters and Validation Dice Coefficient')
plt.show()

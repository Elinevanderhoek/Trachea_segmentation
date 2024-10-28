import numpy as np
import cv2
import os
import skimage
import nibabel as nib
import seaborn as sns
import tensorflow as tf
import warnings
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from scipy.ndimage import binary_fill_holes
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from PIL import Image, ImageDraw
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
warnings.filterwarnings('ignore', category=DeprecationWarning)

def crop_image(image, mask):
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
    mask_round = mask_image[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]

    return img_cropped, mask_round, mask_cropped

def extrapolate(image, mask):
    # Ensure mask is binary
    #mask = (mask > 128).astype(np.uint8)
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

def load_data(loc_img, loc_mask, target_size):
    images = []
    masks = []
    filenames = []
    images_notscaled = []
    masks_notscaled = []

    for file in os.listdir(loc_img):
        f = os.path.join(loc_img, file)
        file_name = os.path.basename(f)
        image = cv2.imread(f, cv2.IMREAD_COLOR)
        images_notscaled.append(image)
        image = cv2.resize(image, target_size)
        images.append(image)
        filenames.append(file_name)

    for file in os.listdir(loc_mask):
        f = os.path.join(loc_mask, file)
        mask = nib.load(f)
        mask = np.array(mask.dataobj)
        mask = mask.transpose(1, 0, 2)
        mask = cv2.resize(mask, image.shape[:2])
        masks_notscaled.append(mask)
        mask = cv2.resize(mask, target_size)
        mask = (mask > 0.5).astype(np.float32)
        masks.append(mask)
    return np.array(images), np.array(masks), filenames, images_notscaled, masks_notscaled

def preprocess_images(images, masks, target_size):
    preprocessed_images = []
    preprocessed_masks = []
    for image, mask in zip(images,masks):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, mask_round, mask_cropped = crop_image(image, mask)
        #image = extrapolate(image, mask_round)
        image = cv2.resize(image, target_size)
        image = np.array(image)/255

        # # Adaptive Equalization
        # image = exposure.equalize_adapthist(image, clip_limit=0.01)
        # image = (image * 255).astype(np.uint8)

        # Apply anisotropic Gaussian blur
        kernel_size = (3, 3) 
        sigma_x = 1
        sigma_y = 1
        image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)

        mask_cropped = cv2.resize(mask_cropped, target_size)

        preprocessed_images.append(image)
        preprocessed_masks.append(mask_cropped)
        
    return np.array(preprocessed_images), np.array(preprocessed_masks)

def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def compute_dice_scores(y_true, y_pred):
    dice_scores = []
    for true, pred in zip(y_true, y_pred):
        # Ensure binary masks
        true = (true > 0).astype(np.float32)
        pred = (pred > 0).astype(np.float32)
        
        # Flatten the masks
        true_f = true.flatten()
        pred_f = pred.flatten()
        
        # Calculate intersection and union
        intersection = np.sum(true_f * pred_f)
        union = np.sum(true_f) + np.sum(pred_f)
        
        # Calculate Dice score
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        dice_scores.append(dice)
    
    return np.array(dice_scores)

def touches_border(contour, width, height):
    return np.any(contour[:, 0, 0] == 0) or np.any(contour[:, 0, 0] == width - 1) or \
           np.any(contour[:, 0, 1] == 0) or np.any(contour[:, 0, 1] == height - 1)

def unet_model():
    inputs = Input(shape=(128, 128, 3))
    kernel_size = 7
    learning_rate = 0.01
    optimizer = RMSprop(learning_rate=learning_rate)  # Set the learning rate here

    # Encoder
    c1 = Conv2D(16, kernel_size, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, kernel_size, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, kernel_size, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(384, kernel_size, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, kernel_size, padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, kernel_size, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, kernel_size, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs=inputs, outputs=[outputs])

    # Compile the model
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])
    return model

def unet_model_with_depth():
    input_size = (128, 128, 3)
    inputs_rgb = Input(input_size, name='input_rgb')
    inputs_depth = Input(input_size, name='input_depth')

    kernel_size = 5

    # Encoder
    c1_rgb = Conv2D(48, kernel_size, padding='same')(inputs_rgb)
    c1_rgb = BatchNormalization()(c1_rgb)
    c1_rgb = Activation('relu')(c1_rgb)
    p1_rgb = MaxPooling2D((2, 2))(c1_rgb)
    
    c2_rgb = Conv2D(64, kernel_size, padding='same')(p1_rgb)
    c2_rgb = BatchNormalization()(c2_rgb)
    c2_rgb = Activation('relu')(c2_rgb)
    p2_rgb = MaxPooling2D((2, 2))(c2_rgb)
    
    c3_rgb = Conv2D(256, kernel_size, padding='same')(p2_rgb)
    c3_rgb = BatchNormalization()(c3_rgb)
    c3_rgb = Activation('relu')(c3_rgb)
    p3_rgb = MaxPooling2D((2, 2))(c3_rgb)

    # Encoder for Depth
    c1_depth = Conv2D(48, kernel_size, padding='same')(inputs_depth)
    c1_depth = BatchNormalization()(c1_depth)
    c1_depth = Activation('relu')(c1_depth)
    p1_depth = MaxPooling2D((2, 2))(c1_depth)
    
    c2_depth = Conv2D(64, kernel_size, padding='same')(p1_depth)
    c2_depth = BatchNormalization()(c2_depth)
    c2_depth = Activation('relu')(c2_depth)
    p2_depth = MaxPooling2D((2, 2))(c2_depth)

    c3_depth = Conv2D(256, kernel_size, padding='same')(p2_depth)
    c3_depth = BatchNormalization()(c3_depth)
    c3_depth = Activation('relu')(c3_depth)
    p3_depth = MaxPooling2D((2, 2))(c3_depth)
    
    merge_bridges = concatenate([p3_rgb, p3_depth])
    # Bottleneck
    c4 = Conv2D(384, kernel_size, padding='same')(merge_bridges)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    u5 = UpSampling2D((2,2))(c4)
    merge5 = concatenate([u5, c3_rgb, c3_depth])
    c5 = Conv2D(256, kernel_size, padding='same')(merge5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2,2))(c5)
    merge6 = concatenate([u6, c2_rgb, c2_depth])
    c6 = Conv2D(64, kernel_size, padding='same')(merge6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2,2))(c6)
    merge7 = concatenate([u7, c1_rgb, c1_depth])
    c7 = Conv2D(48, kernel_size, padding='same')(merge7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs=[inputs_rgb, inputs_depth], outputs=[outputs]) 
    # Compile the model
    sgd_optimizer = SGD(learning_rate=0.01)
    model.compile(optimizer=sgd_optimizer, loss=dice_loss, metrics=[dice_coefficient])
    return model

def unet_model_depth_input():
    inputs = Input(shape=(128, 128, 3))
    kernel_size = 7
    learning_rate = 0.00001
    optimizer = RMSprop(learning_rate=learning_rate)  # Set the learning rate here

    # Encoder
    c1 = Conv2D(48, kernel_size, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, kernel_size, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(192, kernel_size, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, kernel_size, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(192, kernel_size, padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, kernel_size, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(48, kernel_size, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs=inputs, outputs=[outputs])

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
    #image = raw_image/255

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
    #result = cv2.bitwise_and(result,result, mask= mask)

    result_gray = depth_gray.copy()
    #result_gray = cv2.bitwise_and(result_gray,result_gray, mask= mask)

    # plt.figure()
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.title('result')
    # plt.show()

    return result_gray, result_color, final_result

def augment_data(images, masks, augmentation_factor=5):
    augmentation_layer = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation((-0.2,0.2)),
        layers.RandomZoom((0.05,0.2),None),])
    
    augmented_images = []
    augmented_masks = []
    
    for image, mask in zip(images, masks):
        augmented_images.append(image)
        augmented_masks.append(mask)
        
        for _ in range(augmentation_factor - 1):
            # Stack image and mask along the channel dimension
            combined = tf.concat([image, mask], axis=-1)
            augmented_combined = augmentation_layer(combined[np.newaxis, ...], training=True)
            
            # Split the augmented_combined back into image and mask
            augmented_image = augmented_combined[..., :image.shape[-1]]
            augmented_mask = augmented_combined[..., image.shape[-1]:]
            
            augmented_images.append(augmented_image[0])
            augmented_masks.append(augmented_mask[0])
    
    return np.array(augmented_images), np.array(augmented_masks)

def augment_data_multiple(image_arrays, masks, augmentation_factor=5):
    augmentation_layer = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation((-0.2, 0.2)),
        layers.RandomZoom((0.05, 0.2), None),
    ])
    
    augmented_images = [[] for _ in image_arrays]
    augmented_masks = []
    
    for i in range(len(image_arrays[0])):
        images = [arr[i] for arr in image_arrays]
        mask = masks[i]
        
        for j, img in enumerate(images):
            augmented_images[j].append(img)
        augmented_masks.append(mask)
        
        for _ in range(augmentation_factor - 1):
            # Combine all images and mask
            combined = tf.concat([*images, mask], axis=-1)
            augmented_combined = augmentation_layer(combined[np.newaxis, ...], training=True)[0]
            
            # Split the augmented_combined back into images and mask
            split_indices = np.cumsum([img.shape[-1] for img in images])
            augmented_images_split = np.split(augmented_combined[..., :split_indices[-1]], split_indices[:-1], axis=-1)
            augmented_mask = augmented_combined[..., split_indices[-1]:]
            
            for j, aug_image in enumerate(augmented_images_split):
                augmented_images[j].append(aug_image)
            augmented_masks.append(augmented_mask)
    
    return [np.array(aug_images) for aug_images in augmented_images], np.array(augmented_masks)

def visualize_results(images, true_masks, pred_masks, num_examples=3):
    """
    Visualize the original image, ground truth mask, and predicted mask for a given number of examples.
    """
    fig, axes = plt.subplots(num_examples, 3, figsize=(10, 3*num_examples))
    
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

def post_process_prediction(pred_mask):
    # Apply threshold
    binary_mask = (pred_mask.squeeze() > 0.5).astype(np.uint8)

    return binary_mask

def cross_validation(original_images, images, masks, filenames, loc_output, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    dice_scores = []
    all_folds_history = []
    fold_accuracies = []
    pred_masks_all_folds = []

    for fold, (train_index, val_index) in enumerate(kf.split(images)):
        print(f"Output_unet/Training on fold {fold+1}/{n_splits}")
                
        images = np.array(images)
        masks = np.array(masks)
        train_index = np.array(train_index, dtype=int)
        val_index = np.array(val_index, dtype=int)

        X_train = np.array([images[i] for i in train_index])
        X_val = np.array([images[i] for i in val_index])
        filenames_val = np.array([filenames[i] for i in val_index])
        y_train = np.array([masks[i] for i in train_index])
        y_val = np.array([masks[i] for i in val_index])
        X_train, y_train = augment_data(X_train, y_train, augmentation_factor=5)

        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        model = unet_model()
                
        callbacks = [
            ModelCheckpoint(f'Output_unet/best_model_unet_fold_{fold+1}.keras', save_best_only=True, monitor='val_dice_coefficient', mode='max'),
            EarlyStopping(patience=10, monitor='val_dice_coefficient', mode='max'),
            ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.2, patience=5, min_lr=1e-6)
            ]
                
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=35,
            batch_size = 32,
            callbacks=callbacks
            )
        
        print(history.history.keys())
        
        all_folds_history.append(history)

        # Collect the accuracies
        train_accuracy = history.history.get('dice_coefficient', [])[-1]
        val_accuracy = history.history.get('val_dice_coefficient', [])[-1]
        fold_accuracies.append({'fold': fold+1, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy})
                
        # Evaluate on validation set
        val_loss, val_dice = model.evaluate(X_val, y_val)
        fold_scores.append(val_dice)
                
        # Predict on validation set
        pred_masks = model.predict(X_val)
        
        # Optional post-processing
        pred_masks = np.array([post_process_prediction(mask) for mask in pred_masks])
        pred_masks_all_folds.append(pred_masks)
        dice_score = compute_dice_scores(y_val, pred_masks)

        mean_dice = np.mean(dice_score)
        dice_scores.append(mean_dice)
                
        print(f"Fold {fold+1} - Validation Dice: {val_dice}")
        print(f"Fold {fold+1} - Post-processed Validation Dice: {mean_dice}")

        for image, mask, pred_mask, f in zip(X_val, y_val, pred_masks, filenames_val):
            mask = np.squeeze(mask, axis=-1)
            visualize_outcome(image, mask, pred_mask, loc_output, f)
     
        # Visualize results for 5 random examples
        try:
            random_indices = np.random.choice(len(X_val), 5, replace=False)
            #visualize_results(X_val[random_indices], y_val[random_indices], pred_masks[random_indices])
        except Exception as e:
            print(f"Error during visualization: {e}")
        
        # Plot learning curves
        #plot_learning_curves(history, fold)

    epochs = len(all_folds_history[0].history['loss'])

    # Find the maximum length of history among all folds
    max_len = max(len(fold.history['loss']) for fold in all_folds_history)

    # Initialize lists to hold padded/truncated histories
    padded_train_losses = []
    padded_val_losses = []
    padded_train_dice_scores = []
    padded_val_dice_scores = []

    for fold in all_folds_history:
        train_loss = np.array(fold.history['loss'])
        val_loss = np.array(fold.history['val_loss'])
        train_dice = np.array(fold.history['dice_coefficient'])
        val_dice = np.array(fold.history['val_dice_coefficient'])

        # Pad or truncate training loss
        if len(train_loss) < max_len:
            train_loss = np.pad(train_loss, (0, max_len - len(train_loss)), mode='constant', constant_values=np.nan)
        else:
            train_loss = train_loss[:max_len]
        
        # Pad or truncate validation loss
        if len(val_loss) < max_len:
            val_loss = np.pad(val_loss, (0, max_len - len(val_loss)), mode='constant', constant_values=np.nan)
        else:
            val_loss = val_loss[:max_len]

        # Pad or truncate training dice
        if len(train_dice) < max_len:
            train_dice = np.pad(train_dice, (0, max_len - len(train_dice)), mode='constant', constant_values=np.nan)
        else:
            train_dice = train_dice[:max_len]
        
        # Pad or truncate validation dice
        if len(val_dice) < max_len:
            val_dice = np.pad(val_dice, (0, max_len - len(val_dice)), mode='constant', constant_values=np.nan)
        else:
            val_dice = val_dice[:max_len]

        padded_train_losses.append(train_loss)
        padded_val_losses.append(val_loss)
        padded_train_dice_scores.append(train_dice)
        padded_val_dice_scores.append(val_dice)

    # Convert lists to numpy arrays for easier calculations
    padded_train_losses = np.array(padded_train_losses)
    padded_val_losses = np.array(padded_val_losses)
    padded_train_dice_scores = np.array(padded_train_dice_scores)
    padded_val_dice_scores = np.array(padded_val_dice_scores)

    # Calculate the mean and standard deviation across folds for loss
    average_train_loss = np.nanmean(padded_train_losses, axis=0)
    std_train_loss = np.nanstd(padded_train_losses, axis=0)
    average_val_loss = np.nanmean(padded_val_losses, axis=0)
    std_val_loss = np.nanstd(padded_val_losses, axis=0)

    # Calculate the mean and standard deviation across folds for Dice score
    average_train_dice = np.nanmean(padded_train_dice_scores, axis=0)
    std_train_dice = np.nanstd(padded_train_dice_scores, axis=0)
    average_val_dice = np.nanmean(padded_val_dice_scores, axis=0)
    std_val_dice = np.nanstd(padded_val_dice_scores, axis=0)

    # Plot for Loss
    plt.figure()
    epochs = len(average_train_loss)
    plt.plot(range(epochs), average_train_loss, label='Average Training Loss')
    plt.fill_between(range(epochs), average_train_loss - std_train_loss, average_train_loss + std_train_loss, alpha=0.2)
    plt.plot(range(epochs), average_val_loss, label='Average Validation Loss')
    plt.fill_between(range(epochs), average_val_loss - std_val_loss, average_val_loss + std_val_loss, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves Across Folds')
    plt.legend()
    plt.show()

    # Plot for Dice Score
    plt.figure()
    plt.plot(range(epochs), average_train_dice, label='Average Training Dice')
    plt.fill_between(range(epochs), average_train_dice - std_train_dice, average_train_dice + std_train_dice, alpha=0.2)
    plt.plot(range(epochs), average_val_dice, label='Average Validation Dice')
    plt.fill_between(range(epochs), average_val_dice - std_val_dice, average_val_dice + std_val_dice, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Dice Score Across Folds')
    plt.legend()
    plt.show()

    avg_dice_score = np.mean(dice_scores)
    pred_masks_all_folds = [item for sublist in pred_masks_all_folds for item in sublist]

    return avg_dice_score, pred_masks, X_val, pred_masks_all_folds

def train_final_unet_model(original_images, images, masks, filenames, loc_output):
    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)

    # Augment the entire dataset
    X_train, y_train = augment_data(images, masks, augmentation_factor=5)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training masks shape: {y_train.shape}")

    # Create and compile the model
    model = unet_model()

    # Define callbacks
    callbacks = [
        ModelCheckpoint(f'{loc_output}/best_model_unet_final.keras', save_best_only=True, monitor='dice_coefficient', mode='max'),
        EarlyStopping(patience=10, monitor='dice_coefficient', mode='max'),
        ReduceLROnPlateau(monitor='dice_coefficient', factor=0.2, patience=5, min_lr=1e-6)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=35,
        batch_size=32,
        callbacks=callbacks
    )

    print(history.history.keys())

    # Plot learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Training Dice Coefficient')
    plt.title('Training Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{loc_output}/learning_curves_final_model.png')
    plt.close()

    # Predict on the entire dataset
    pred_masks = model.predict(images)
    pred_masks = np.array([post_process_prediction(mask) for mask in pred_masks])

    # Compute Dice score
    dice_score = compute_dice_scores(masks, pred_masks)
    mean_dice = np.mean(dice_score)
    print(f"Final model - Dice Score: {mean_dice}")

    # Visualize outcomes for all images
    for image, mask, pred_mask, f in zip(images, masks, pred_masks, filenames):
        mask = np.squeeze(mask, axis=-1)
        visualize_outcome(image, mask, pred_mask, loc_output, f)

    return model, pred_masks, mean_dice

def cross_validation_depth_input(images, masks, preprocessed_images, filenames, loc_output, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    dice_scores = []
    all_folds_history = []
    fold_accuracies = []
    pred_masks_all_folds = []

    for fold, (train_index, val_index) in enumerate(kf.split(images)):
        print(f"Training on fold {fold+1}/{n_splits}")

        images = np.array(images)
        masks = np.array(masks)
        train_index = np.array(train_index, dtype=int)
        val_index = np.array(val_index, dtype=int)

        X_train = np.array([images[i] for i in train_index])
        X_val = np.array([images[i] for i in val_index])
        X_val_original = np.array([preprocessed_images[i] for i in val_index])
        filenames_val = np.array([filenames[i] for i in val_index])
        y_train = np.array([masks[i] for i in train_index])
        y_val = np.array([masks[i] for i in val_index])

        X_train, y_train = augment_data(X_train, y_train, augmentation_factor=5)

        model = unet_model_depth_input()
                
        callbacks = [
            ModelCheckpoint(f'Output_unet_depth_input/best_model_depth_input_fold_{fold+1}.keras', save_best_only=True, monitor='val_dice_coefficient', mode='max'),
            EarlyStopping(patience=10, monitor='val_dice_coefficient', mode='max'),
            ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.2, patience=5, min_lr=1e-6)
            ]
                
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=45,
            batch_size = 24,
            callbacks=callbacks
            )
        
        print(history.history.keys())
        
        all_folds_history.append(history)

        # Collect the accuracies
        train_accuracy = history.history.get('dice_coefficient', [])[-1]
        val_accuracy = history.history.get('val_dice_coefficient', [])[-1]
        fold_accuracies.append({'fold': fold+1, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy})
                
        # Evaluate on validation set
        val_loss, val_dice = model.evaluate(X_val, y_val)
        fold_scores.append(val_dice)
                
        # Predict on validation set
        pred_masks = model.predict(X_val)
        
        # Optional post-processing
        pred_masks = np.array([post_process_prediction(mask) for mask in pred_masks])
        pred_masks_all_folds.append(pred_masks)
        dice_score = compute_dice_scores(y_val, pred_masks)

        mean_dice = np.mean(dice_score)
        dice_scores.append(mean_dice)
                
        print(f"Fold {fold+1} - Validation Dice: {val_dice}")
        print(f"Fold {fold+1} - Post-processed Validation Dice: {mean_dice}")

        target_shape = (1024, 1024, 3)  # Define your target shape

        for image, mask, pred_mask, f in zip(X_val_original, y_val, pred_masks, filenames_val):
            mask = np.squeeze(mask, axis=-1)
            if image.max() == 1.0:
                image = (image * 255).astype(np.uint8)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            visualize_outcome(image, mask, pred_mask, loc_output, f)
     
        # Visualize results for 5 random examples
        try:
            random_indices = np.random.choice(len(X_val), 5, replace=False)
            #visualize_results(X_val[random_indices], y_val[random_indices], pred_masks[random_indices])
        except Exception as e:
            print(f"Error during visualization: {e}")
        
        # Plot learning curves
        #plot_learning_curves(history, fold)

    epochs = len(all_folds_history[0].history['loss'])

    # Find the maximum length of history among all folds
    max_len = max(len(fold.history['loss']) for fold in all_folds_history)

    # Initialize lists to hold padded/truncated histories
    padded_train_losses = []
    padded_val_losses = []
    padded_train_dice_scores = []
    padded_val_dice_scores = []

    for fold in all_folds_history:
        train_loss = np.array(fold.history['loss'])
        val_loss = np.array(fold.history['val_loss'])
        train_dice = np.array(fold.history['dice_coefficient'])  # Assuming 'dice_coefficient' is in the history
        val_dice = np.array(fold.history['val_dice_coefficient'])  # Assuming 'val_dice_coefficient' is in the history

        # Pad or truncate training loss
        if len(train_loss) < max_len:
            train_loss = np.pad(train_loss, (0, max_len - len(train_loss)), mode='constant', constant_values=np.nan)
        else:
            train_loss = train_loss[:max_len]
        
        # Pad or truncate validation loss
        if len(val_loss) < max_len:
            val_loss = np.pad(val_loss, (0, max_len - len(val_loss)), mode='constant', constant_values=np.nan)
        else:
            val_loss = val_loss[:max_len]

        # Pad or truncate training dice
        if len(train_dice) < max_len:
            train_dice = np.pad(train_dice, (0, max_len - len(train_dice)), mode='constant', constant_values=np.nan)
        else:
            train_dice = train_dice[:max_len]
        
        # Pad or truncate validation dice
        if len(val_dice) < max_len:
            val_dice = np.pad(val_dice, (0, max_len - len(val_dice)), mode='constant', constant_values=np.nan)
        else:
            val_dice = val_dice[:max_len]

        padded_train_losses.append(train_loss)
        padded_val_losses.append(val_loss)
        padded_train_dice_scores.append(train_dice)
        padded_val_dice_scores.append(val_dice)

    # Convert lists to numpy arrays for easier calculations
    padded_train_losses = np.array(padded_train_losses)
    padded_val_losses = np.array(padded_val_losses)
    padded_train_dice_scores = np.array(padded_train_dice_scores)
    padded_val_dice_scores = np.array(padded_val_dice_scores)

    # Calculate the mean and standard deviation across folds for loss
    average_train_loss = np.nanmean(padded_train_losses, axis=0)
    std_train_loss = np.nanstd(padded_train_losses, axis=0)
    average_val_loss = np.nanmean(padded_val_losses, axis=0)
    std_val_loss = np.nanstd(padded_val_losses, axis=0)

    # Calculate the mean and standard deviation across folds for Dice score
    average_train_dice = np.nanmean(padded_train_dice_scores, axis=0)
    std_train_dice = np.nanstd(padded_train_dice_scores, axis=0)
    average_val_dice = np.nanmean(padded_val_dice_scores, axis=0)
    std_val_dice = np.nanstd(padded_val_dice_scores, axis=0)

    # Plot for Loss
    plt.figure()
    epochs = len(average_train_loss)
    plt.plot(range(epochs), average_train_loss, label='Average Training Loss')
    plt.fill_between(range(epochs), average_train_loss - std_train_loss, average_train_loss + std_train_loss, alpha=0.2)
    plt.plot(range(epochs), average_val_loss, label='Average Validation Loss')
    plt.fill_between(range(epochs), average_val_loss - std_val_loss, average_val_loss + std_val_loss, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves Across Folds')
    plt.legend()
    plt.show()

    # Plot for Dice Score
    plt.figure()
    plt.plot(range(epochs), average_train_dice, label='Average Training Dice')
    plt.fill_between(range(epochs), average_train_dice - std_train_dice, average_train_dice + std_train_dice, alpha=0.2)
    plt.plot(range(epochs), average_val_dice, label='Average Validation Dice')
    plt.fill_between(range(epochs), average_val_dice - std_val_dice, average_val_dice + std_val_dice, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Dice Score Across Folds')
    plt.legend()
    plt.show()

    avg_dice_score = np.mean(dice_scores)
    pred_masks_all_folds = [item for sublist in pred_masks_all_folds for item in sublist]
    return avg_dice_score, pred_masks, pred_masks_all_folds

def train_final_model_depth_input(depth_images, masks, filenames, loc_output):
    # # Convert lists to numpy arrays
    # images = np.array(images)
    # masks = np.array(masks)

    # Augment the entire dataset
    X_train, y_train = augment_data(depth_images, masks, augmentation_factor=5)

    # Create and compile the model
    model = unet_model_depth_input()

    # Define callbacks
    callbacks = [
        ModelCheckpoint(f'{loc_output}/best_model_depth_input_final.keras', save_best_only=True, monitor='dice_coefficient', mode='max'),
        EarlyStopping(patience=10, monitor='dice_coefficient', mode='max'),
        ReduceLROnPlateau(monitor='dice_coefficient', factor=0.2, patience=5, min_lr=1e-6)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=45,
        batch_size=24,
        callbacks=callbacks
    )

    # Plot learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Training Dice Coefficient')
    plt.title('Training Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{loc_output}/learning_curves_final_model.png')
    plt.close()

    # Predict on the entire dataset
    pred_masks = model.predict(depth_images)
    pred_masks = np.array([post_process_prediction(mask) for mask in pred_masks])

    # Compute Dice score
    dice_score = compute_dice_scores(masks, pred_masks)
    mean_dice = np.mean(dice_score)
    print(f"Final model - Dice Score: {mean_dice}")

    return model, pred_masks, mean_dice

def cross_validation_depth(original_images, rgb_images, depth_images, masks, filenames, loc_output, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    dice_scores = []
    all_folds_history = []
    fold_accuracies = []
    pred_masks_all_folds = []

    for fold, (train_index, val_index) in enumerate(kf.split(rgb_images)):
        print(f"Training on fold {fold+1}/{n_splits}")

        rgb_images = np.array(rgb_images)
        depth_images = np.array(depth_images)
        masks = np.array(masks)
        train_index = np.array(train_index, dtype=int)
        val_index = np.array(val_index, dtype=int)

        X_train_rgb = np.array([rgb_images[i] for i in train_index])
        X_val_rgb = np.array([rgb_images[i] for i in val_index])
        X_train_depth = np.array([depth_images[i] for i in train_index])
        X_val_depth = np.array([depth_images[i] for i in val_index])
        filenames_val = np.array([filenames[i] for i in val_index])
        y_train = np.array([masks[i] for i in train_index])
        y_val = np.array([masks[i] for i in val_index])

        # Augment data
        (X_train_rgb, X_train_depth), y_train = augment_data_multiple(
            [X_train_rgb, X_train_depth], y_train, augmentation_factor=5)

        # Combine RGB and Depth data
        X_train = [np.array(X_train_rgb), np.array(X_train_depth)]
        X_val = [np.array(X_val_rgb), np.array(X_val_depth)]

        model = unet_model_with_depth()  # Make sure this function is defined to handle both RGB and depth inputs
                
        callbacks = [
            ModelCheckpoint(f'Output_unet_depth_combi/best_model_with_depth_fold_{fold+1}.keras', save_best_only=True, monitor='val_dice_coefficient', mode='max'),
            EarlyStopping(patience=10, monitor='val_dice_coefficient', mode='max'),
            ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.2, patience=5, min_lr=1e-6)
        ]
                
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=8,
            callbacks=callbacks
        )
        
        print(history.history.keys())
        
        all_folds_history.append(history)

        # Collect the accuracies
        train_accuracy = history.history.get('dice_coefficient', [])[-1]
        val_accuracy = history.history.get('val_dice_coefficient', [])[-1]
        fold_accuracies.append({'fold': fold+1, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy})
                
        # Evaluate on validation set
        val_loss, val_dice = model.evaluate(X_val, y_val)
        fold_scores.append(val_dice)
                
        # Predict on validation set
        pred_masks = model.predict(X_val)
        
        # Optional post-processing
        pred_masks = np.array([post_process_prediction(mask) for mask in pred_masks])
        pred_masks_all_folds.append(pred_masks)
        dice_score = compute_dice_scores(y_val, pred_masks)

        mean_dice = np.mean(dice_score)
        dice_scores.append(mean_dice)
                
        print(f"Fold {fold+1} - Validation Dice: {val_dice}")
        print(f"Fold {fold+1} - Post-processed Validation Dice: {mean_dice}")

        # Visualization
        for image, mask, pred_mask, f in zip(X_val[0], y_val, pred_masks, filenames_val):
            mask = np.squeeze(mask, axis=-1)
            if image.max() == 1.0:
                image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            visualize_outcome(image, mask, pred_mask, loc_output, f)

    # Visualize results for 5 random examples
        try:
            random_indices = np.random.choice(len(X_val), 5, replace=False)
            #visualize_results(X_val[random_indices], y_val[random_indices], pred_masks[random_indices])
        except Exception as e:
            print(f"Error during visualization: {e}")
        
        # Plot learning curves
        #plot_learning_curves(history, fold)

    epochs = len(all_folds_history[0].history['loss'])

    # Find the maximum length of history among all folds
    max_len = max(len(fold.history['loss']) for fold in all_folds_history)

    # Initialize lists to hold padded/truncated histories
    padded_train_losses = []
    padded_val_losses = []
    padded_train_dice_scores = []
    padded_val_dice_scores = []

    for fold in all_folds_history:
        train_loss = np.array(fold.history['loss'])
        val_loss = np.array(fold.history['val_loss'])
        train_dice = np.array(fold.history['dice_coefficient'])  # Assuming 'dice_coefficient' is in the history
        val_dice = np.array(fold.history['val_dice_coefficient'])  # Assuming 'val_dice_coefficient' is in the history

        # Pad or truncate training loss
        if len(train_loss) < max_len:
            train_loss = np.pad(train_loss, (0, max_len - len(train_loss)), mode='constant', constant_values=np.nan)
        else:
            train_loss = train_loss[:max_len]
        
        # Pad or truncate validation loss
        if len(val_loss) < max_len:
            val_loss = np.pad(val_loss, (0, max_len - len(val_loss)), mode='constant', constant_values=np.nan)
        else:
            val_loss = val_loss[:max_len]

        # Pad or truncate training dice
        if len(train_dice) < max_len:
            train_dice = np.pad(train_dice, (0, max_len - len(train_dice)), mode='constant', constant_values=np.nan)
        else:
            train_dice = train_dice[:max_len]
        
        # Pad or truncate validation dice
        if len(val_dice) < max_len:
            val_dice = np.pad(val_dice, (0, max_len - len(val_dice)), mode='constant', constant_values=np.nan)
        else:
            val_dice = val_dice[:max_len]

        padded_train_losses.append(train_loss)
        padded_val_losses.append(val_loss)
        padded_train_dice_scores.append(train_dice)
        padded_val_dice_scores.append(val_dice)

    # Convert lists to numpy arrays for easier calculations
    padded_train_losses = np.array(padded_train_losses)
    padded_val_losses = np.array(padded_val_losses)
    padded_train_dice_scores = np.array(padded_train_dice_scores)
    padded_val_dice_scores = np.array(padded_val_dice_scores)

    # Calculate the mean and standard deviation across folds for loss
    average_train_loss = np.nanmean(padded_train_losses, axis=0)
    std_train_loss = np.nanstd(padded_train_losses, axis=0)
    average_val_loss = np.nanmean(padded_val_losses, axis=0)
    std_val_loss = np.nanstd(padded_val_losses, axis=0)

    # Calculate the mean and standard deviation across folds for Dice score
    average_train_dice = np.nanmean(padded_train_dice_scores, axis=0)
    std_train_dice = np.nanstd(padded_train_dice_scores, axis=0)
    average_val_dice = np.nanmean(padded_val_dice_scores, axis=0)
    std_val_dice = np.nanstd(padded_val_dice_scores, axis=0)

    # Plot for Loss
    plt.figure()
    epochs = len(average_train_loss)
    plt.plot(range(epochs), average_train_loss, label='Average Training Loss')
    plt.fill_between(range(epochs), average_train_loss - std_train_loss, average_train_loss + std_train_loss, alpha=0.2)
    plt.plot(range(epochs), average_val_loss, label='Average Validation Loss')
    plt.fill_between(range(epochs), average_val_loss - std_val_loss, average_val_loss + std_val_loss, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves Across Folds')
    plt.legend()
    plt.show()

    # Plot for Dice Score
    plt.figure()
    plt.plot(range(epochs), average_train_dice, label='Average Training Dice')
    plt.fill_between(range(epochs), average_train_dice - std_train_dice, average_train_dice + std_train_dice, alpha=0.2)
    plt.plot(range(epochs), average_val_dice, label='Average Validation Dice')
    plt.fill_between(range(epochs), average_val_dice - std_val_dice, average_val_dice + std_val_dice, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Dice Score Across Folds')
    plt.legend()
    plt.show()

    avg_dice_score = np.mean(dice_scores)
    pred_masks_all_folds = [item for sublist in pred_masks_all_folds for item in sublist]
    return avg_dice_score, pred_masks, pred_masks_all_folds

def train_final_model_depth(original_images, rgb_images, depth_images, masks, filenames, loc_output):
    # Convert lists to numpy arrays
    rgb_images = np.array(rgb_images)
    depth_images = np.array(depth_images)
    masks = np.array(masks)

    # Augment the entire dataset
    (X_train_rgb, X_train_depth), y_train = augment_data_multiple(
        [rgb_images, depth_images], masks, augmentation_factor=5)

    # Combine RGB and Depth data
    X_train = [np.array(X_train_rgb), np.array(X_train_depth)]

    # Create and compile the model
    model = unet_model_with_depth()

    # Define callbacks
    callbacks = [
        ModelCheckpoint(f'{loc_output}/best_model_with_depth_final.keras', save_best_only=True, monitor='dice_coefficient', mode='max'),
        EarlyStopping(patience=10, monitor='dice_coefficient', mode='max'),
        ReduceLROnPlateau(monitor='dice_coefficient', factor=0.2, patience=5, min_lr=1e-6)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=8,
        callbacks=callbacks
    )

    # Plot learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Training Dice Coefficient')
    plt.title('Training Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{loc_output}/learning_curves_final_model_with_depth.png')
    plt.close()

    # Predict on the entire dataset
    X_predict = [rgb_images, depth_images]
    pred_masks = model.predict(X_predict)
    pred_masks = np.array([post_process_prediction(mask) for mask in pred_masks])

    # Compute Dice score
    dice_score = compute_dice_scores(masks, pred_masks)
    mean_dice = np.mean(dice_score)
    print(f"Final model - Dice Score: {mean_dice}")

    # # Visualization for a few samples
    # num_samples = min(5, len(rgb_images))
    # for i in range(num_samples):
    #     image = rgb_images[i]
    #     mask = masks[i]
    #     pred_mask = pred_masks[i]
    #     filename = filenames[i]

    #     mask = np.squeeze(mask, axis=-1)
    #     if image.max() == 1.0:
    #         image = (image * 255).astype(np.uint8)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     visualize_outcome(image, mask, pred_mask, loc_output, f'final_model_{filename}')

    return model, pred_masks, mean_dice

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def calc_diam_area(segment, image, f, loc_output):
    image_scaled = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    #image_scaled = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
    segment = (segment > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(segment.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Take the largest contour
    largest_contour = cnts[0]
    box = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(box)  # Use cv2.boxPoints for recent OpenCV versions
    box = np.array(box, dtype="int")

	# Order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order
    box = perspective.order_points(box)
    
    # Draw the outline of the rotated bounding box
    cv2.drawContours(segment, [box.astype("int")], -1, (0, 255, 0), 2)
	
	# loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(segment, (int(x), int(y)), 5, (0, 0, 255), -1)
	
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
	
	# compute the midpoint between the top-left and top-right points, followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
	
    final2 = image_scaled.copy()
	# draw the midpoints on the image
    cv2.circle(final2, (int(tltrX), int(tltrY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)
	
	# draw lines between the midpoints
    cv2.line(final2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(final2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

    base_name = os.path.splitext(f)[0]  # Get the base name without extension
    
    save_path = f"{loc_output}/{base_name}_dimensions.png"
    cv2.imwrite(save_path, final2)  # Save the image
	
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    perimeter = 0
    for contour in cnts:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

    return dA, dB, area, perimeter, final2

def calc_diam_area2(segment, image, f, loc_output):
    image_scaled = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    image_scaled = cv2.cvtColor(image_scaled, cv2.COLOR_RGB2BGR)
    segment = segment.astype(np.uint8)
    cnts, _ = cv2.findContours(segment.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Take the largest contour
    largest_contour = cnts[0]
    box = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(box)  # Use cv2.boxPoints for recent OpenCV versions
    box = np.array(box, dtype="int")

	# Order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order
    box = perspective.order_points(box)
    
    # Draw the outline of the rotated bounding box
    cv2.drawContours(segment, [box.astype("int")], -1, (0, 255, 0), 2)
	
	# loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(segment, (int(x), int(y)), 5, (0, 0, 255), -1)
	
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
	
	# compute the midpoint between the top-left and top-right points, followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
	
    final2 = image_scaled.copy()
	# draw the midpoints on the image
    cv2.circle(final2, (int(tltrX), int(tltrY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)
	
	# draw lines between the midpoints
    cv2.line(final2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(final2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

    base_name = os.path.splitext(f)[0]  # Get the base name without extension
    
    save_path = f"{loc_output}/{base_name}_dimensions_ground_truth.png"
    cv2.imwrite(save_path, final2)  # Save the image
	
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    perimeter = 0
    for contour in cnts:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

    return dA, dB, area, perimeter, final2

def visualize_outcome(image, mask, pred_mask, loc_output, f):
    # Ensure masks are binary
    mask = (mask > 0.5).astype(np.uint8)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Define a kernel for dilation
    kernel = np.ones((3, 3), np.uint8)
    
    # Dilate the ground truth mask and subtract the original to get the boundary
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    boundary_mask = dilated_mask - mask
    
    # Dilate the predicted mask and subtract the original to get the boundary
    dilated_pred_mask = cv2.dilate(pred_mask, kernel, iterations=1)
    boundary_pred_mask = dilated_pred_mask - pred_mask
    
    # Ensure boundary masks have the same shape as the image
    if boundary_mask.shape != (image.shape[0], image.shape[1]):
        raise ValueError("Boundary mask dimensions must match image dimensions.")

    # Create color boundaries: green for ground truth, red for prediction
    image_with_boundaries = image.astype(np.uint8)
    image_with_boundaries = cv2.cvtColor(image_with_boundaries, cv2.COLOR_BGR2RGB)
    image_with_boundaries[boundary_mask > 0] = [0, 255, 0]  # Green boundary for ground truth
    image_with_boundaries[boundary_pred_mask > 0] = [255, 0, 0]  # Red/blue boundary for prediction

    base_name = os.path.splitext(f)[0]  # Get the base name without extension
    
    save_path = f"{loc_output}/{base_name}_final.png"
    
    cv2.imwrite(save_path, image_with_boundaries)  # Save the image

    # Optionally, you can uncomment the following lines to display the image
    # plt.figure(figsize=(12, 6) if depth_image is not None else (8, 8))
    # plt.imshow(combined_image)
    # plt.title('Image with Ground Truth and Prediction Boundaries' + (' and Depth' if depth_image is not None else ''))
    # plt.axis('off')
    # plt.show()
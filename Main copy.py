import cv2
import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename
import math
import matplotlib.pyplot as plt
import re
from pathlib import Path
import Canny_edge
import copy_runV2_copy
import threshold_kmeans
import Extrapolate
import threshold_hist
import create_hist
import pointcloud
import fill_oval_contours
import Calc_diam_area
import find_k_optimal
import os
import warnings
import ctypes
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Point cloud works on python 3.11 versions (or older), because of open3d

folder = 'Frames' #important: frame has to be sharp and the camera has to be in the middle of the trachea
image_path = Path(folder).glob('*.png')
all_files = os.listdir('./Frames/')
file_names = []

ctypes.windll.user32.MessageBoxW(0, "In the next window, choose your inspiratory (or normal) frame. In the one thereafter, choose your expiratory (or malacia) frame.", "Choose your frame", 0)
filename1 = askopenfilename(title = 'Choose your inspiratory (or normal) frame')
filename2 = askopenfilename(title = 'Choose your expiratory (or malacia) frame')
file_names.append(str(filename1))
file_names.append(str(filename2))

dataframe = pd.DataFrame(columns=['Sharpness', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio dA/dB', 'Ratio P/A'])
rows = []
indices = []
names = []

for file in file_names: 
    image = cv2.imread(file)
    pattern = '[\w-]+?(?=\.)'
    name = re.search(pattern, file)
    name = name.group()
    print(name)

    canny = cv2.Canny(image, 50,250)

    if np.mean(canny) > 1.0:
        sigma = (6,6)
        sigma2 = 1
        im_gray, result, gray, cropped_color, mask = Canny_edge.mask(image, sigma, sigma2)
        result2 = Extrapolate.extrapolate(result, mask)

        depth_gray, depth_color, final_result = copy_runV2_copy.depth_anything_function(result2)
        #scaled_hist, histr, histr_gray, image, gray = create_hist.create_hist_func(depth_color, depth_gray)
        #pointcloud.point_cloud(gray, depth_gray)

        #Thresholding with kmeans
        n_clusters = find_k_optimal.find_k_optimal(depth_color)
        reduced, mask_image = threshold_kmeans.thresholding_algo_kmeans(n_clusters, depth_color, result, gray)

        #Thresholding with histogram peaks
        #prominance = 60
        #result2 = threshold_hist.thresholding_algo_hist(scaled_hist, gray, prominance, result)

        contour, final, segment = fill_oval_contours.fill_oval_contours(mask_image, result)

        dA, dB, area, peri = Calc_diam_area.calc_diam_area(contour, final)
        ratio = dA/dB
        ratio2 = 4*math.pi*area/(peri**2) #calculate roundness
        rows.append({'Sharpness': np.mean(canny), 'dA': dA, 'dB': dB, 'Area': area, 'Perimeter': peri, 'Ratio dA/dB': ratio, 'Ratio P/A': ratio2})
        indices.append(name)
        print('')
    else:
        print('Image resolution too low')
        indices.append(name)
        rows.append({'Sharpness': np.mean(canny), 'dA': '?', 'dB': '?', 'Area': '?', 'Perimeter': '?', 'Ratio dA/dB': '?', 'Ratio P/A': '?'})
        continue

dataframe = pd.DataFrame(rows, columns=['Sharpness', 'dA', 'dB', 'Area', 'Perimeter', 'Ratio dA/dB', 'Ratio P/A'])
dataframe.index = indices
print(dataframe)

area_insp = dataframe.loc[[indices[0]], ['Area']].values
area_exp = dataframe.loc[[indices[1]],["Area"]].values
area_perc = 100 - (area_exp / area_insp * 100)
print('')
print(f'Percentage of collaps based on area:', int(area_perc))

ratio_insp = dataframe.loc[[indices[0]], ['Ratio dA/dB']].values *100
ratio_exp = dataframe.loc[[indices[1]], ["Ratio dA/dB"]].values *100
ratio_perc = 100 - (ratio_exp / ratio_insp * 100)
print(f'Percentage of collaps based on diameters ratios:', int(ratio_perc))

ratio2_insp = dataframe.loc[[indices[0]], ['Ratio P/A']].values *100
ratio2_exp = dataframe.loc[[indices[1]], ["Ratio P/A"]].values *100
ratio2_perc = (100 - (ratio2_exp / ratio2_insp * 100)) 
print(f'Percentage change of roundness:', int(ratio2_perc))

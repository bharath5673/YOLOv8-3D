
from tensorflow.keras import backend as K
from IPython.display import SVG
import matplotlib.pyplot as plt

import copy
import cv2, os
import numpy as np
from random import shuffle
import pandas as pd


import tqdm
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage

import sys
sys.path.append('../')
from train import *


# Define the augmentation sequence
seq = iaa.Sequential([
    # iaa.Crop(px=(0, 7)),  # crop images from each side by 0 to 16px
    iaa.Crop(px=(7, 0)),  # crop images from each side by 0 to 16px
    # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
])




if __name__ == '__main__':
    all_exams  = len(all_objs)
    trv_split  = int(0.85*all_exams)
    batch_size = BATCH_SIZE
    np.random.shuffle(all_objs)

    train_gen = data_gen(all_objs[:trv_split],          batch_size)
    valid_gen = data_gen(all_objs[trv_split:all_exams], batch_size)

    train_num = int(np.ceil(trv_split/batch_size))
    valid_num = int(np.ceil((all_exams - trv_split)/batch_size))


    # Assuming train_gen yields batches of tuples (images, annotations)
    images, annotations = next(train_gen)
    image = images[0]
    annotation = annotations[0]




    # Extract only the first 2 columns (x and y coordinates) to get 2D keypoints
    keypoints_2d = annotation[:, :2]
    keypoints = KeypointsOnImage.from_xy_array(keypoints_2d, shape=image.shape)
    augmented_image, augmented_keypoint = seq(images=[image], keypoints=keypoints)


    # 'augmented_images' will contain the augmented images, and 'augmented_annotations' will have corresponding annotations
    # print(np.array(augmented_keypoint))
    augmented_annotation_array = augmented_keypoint.to_xy_array()


    # Extract x, y coordinates for original and augmented annotations
    x1 = annotation[:, 0]
    y1 = annotation[:, 1]
    x2 = augmented_annotation_array[:, 0]
    y2 = augmented_annotation_array[:, 1]

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the original image and annotations
    axes[0].imshow(image)
    axes[0].scatter(x1, y1, color='blue', label='Annotations')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Original Image')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend()

    # Plot the augmented image and annotations
    axes[1].imshow(augmented_image[0])
    axes[1].scatter(x2, y2, color='blue', label='Annotations')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Augmented Image')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()



    ### augmentations
    # ls = []
    # ls2 = []
    # # Assuming train_gen yields batches of tuples (images, annotations)
    # for images, annotations in tqdm.tqdm(train_gen):
    #     for image, annotation in zip(images, annotations):
    #         ls.append([image, annotation])

    #         # Extract only the first 2 columns (x and y coordinates) to get 2D keypoints
    #         keypoints_2d = annotation[:, :2]
    #         assert keypoints_2d.shape[1] == 2, "Keypoints should have shape (N, 2)"
    #         keypoints_2d = keypoints_2d.reshape(-1, 2)
    #         keypoints = KeypointsOnImage.from_xy_array(keypoints_2d, shape=image.shape)
    #         augmented_image, augmented_keypoint = seq(images=[image], keypoints=keypoints)
    #         ls2.append([augmented_image, annotation])

    # ls3 = ls + ls2
    # print(len(ls3))

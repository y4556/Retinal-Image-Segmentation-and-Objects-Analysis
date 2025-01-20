# Retinal-Image-Segmentation-and-Objects-Analysis
<img src="https://github.com/y4556/Retinal-Image-Segmentation-and-Objects-Analysis/blob/main/RetinaImage.JPG?raw=true" alt="Retinal Image" width="300"/>


## Overview  
This project implements a series of algorithms for analyzing fundus images to locate the optic disc and evaluate its characteristics. The pipeline includes preprocessing, vein structure enhancement, optic disc localization using various methods, and error analysis to assess the accuracy of the detected features.  

## Objective 
The project aims to:
- Extract the optic disc from given fundus images using image processing techniques.
- Identify the true optic disc region amidst bright lesions.
- Calculate and report localization errors using Euclidean distance.

## Dataset 
The dataset includes 50 fundus images, each with:
- Original colored fundus image.
- Corresponding blood vessel map.
- Ground truth coordinates of the optic disc location.
[Download the Dataset](https://drive.google.com/drive/folders/1DxmL9I2772qTCYwlbMk1KpKPtJb85o-H)
 

## Features
1. **Preprocessing**:
   - Enhances vein structures using morphological operations and adaptive thresholding.

2. **Vein Mask Generation**:
   - Extracts a binary mask highlighting the veins in grayscale fundus images.

3. **Feature Extraction**:
   - Identifies potential optic disc locations using sum filters and brightest pixel detection.

4. **Error Analysis**:
   - Computes the Euclidean distance between the detected optic disc center and the ground truth.

5. **Visualization**:
   - Overlays detected features on the original images and saves the resulting outputs for analysis.

## Installation
1. Clone the repository:
   ```bash
   [git clone https://github.com/yourusername/fundus-image-analysis.git
   cd fundus-image-analysis](https://github.com/y4556/Retinal-Image-Segmentation-and-Objects-Analysis/tree/main)

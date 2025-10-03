# Identication-Corn-Leaf-Disease
This repository contains a research project on corn leaf disease identification using image processing techniques. The project was developed as part of my undergraduate thesis at the University of Jember (Physics, Electronics & Instrumentation). The goal of this study is to provide early detection of corn leaf diseases through image-based classification.

---

Corn is one of the most widely cultivated crops, but its productivity can be reduced by various leaf diseases. Early detection is crucial to prevent production loss. This project applies Gray Level Co-occurrence Matrix (GLCM) for texture feature extraction and K-Nearest Neighbors (K-NN) for classification of corn leaf diseases.

---

Methods
1. Dataset
   - Training data: 1200 images (from Kaggle).
   - Testing data: 280 images (collected directly in Wringinagung, Jember, East Java using iPhone 13 camera).
   - 4 disease classes:
     - Healthy Leaf
     - Leaf Blight
     - Leaf Rust
     - Gray Leaf Spot

2. Preprocessing
   - Cropping (manual)
   - Resizing (450 × 450 px)
   - Grayscaling (RGB → grayscale)

3. Feature Extraction (GLCM)
   - Calculated parameters: Contrast, Correlation, Homogeneity, Energy
   - Angles: 0°, 45°, 90°, and 135°

4. Classification (K-NN)
   - Euclidean Distance used as distance metric
   - Various values of *k* tested (1–16)

---

Results
- Modified GLCM + K-NN achieved better performance compared to MATLAB built-in functions.
- Best accuracy: 62.50% (k = 1)
- Worst accuracy: 43.57% (k = 16)
- A simple Graphical User Interface (GUI) was implemented to allow users to classify new images easily.

---

GUI Example
The GUI provides buttons to:
- Upload corn leaf images
- Perform training and testing
- Display classification results

---

Tech Stack
- MATLAB (image preprocessing, feature extraction, classification)
- Image dataset (Kaggle + field collection)
- Basic GUI implementation in MATLAB

---

Author
Adelia Indah Wahyuni  
 

---

## 🔗 Notes
- Dataset from Kaggle is not included due to size restrictions.  
- Testing dataset (280 field images) available upon request. 

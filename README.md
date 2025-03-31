# Classifier Model for Garment Classification

## 1. Introduction
The goal of this project was to train a deep learning model to classify clothing items into categories such as **Upper Garment**, **Lower Garment**, and other types. We aimed to structure classifications hierarchically by **gender** (e.g., men, women), **type** (e.g., shirt, pants), and **subcategories** (e.g., casual, formal).

To achieve this, we utilized a pre-trained deep learning model and adapted it for improved classification performance.

---

## 2. Model Selection and Adaptation

We based our implementation on the GitHub repository: **[FarnooshGhadiri/Cloth_category_classifier](https://github.com/FarnooshGhadiri/Cloth_category_classifier)**. This model uses **ResNet-50** as a backbone for feature extraction.

### **Model Architecture**
The custom `ClothingClassifier` includes:
- A **ResNet-50** backbone for robust feature extraction.
- A **fully connected layer** to reduce the dimensionality to 512.
- A **presence classifier** to determine whether **topwear**, **bottomwear**, or **both** are present.
- Three dedicated classifiers for:
  - Topwear categories
  - Bottomwear categories
  - Full-body garment categories (e.g., dresses)

---

## 3. Training Strategy

### **Hyperparameter Tuning**
We performed **hyperparameter tuning** to determine the optimal number of frozen layers in the ResNet-50 backbone. This helped balance between retaining pretrained knowledge and adapting to the new task, improving convergence and generalization.

### **Handling Class Imbalance**
To address class imbalance:
- We used a **weighted cross-entropy loss function**, giving more weight to minority classes to prevent the model from overfitting to dominant categories.
- We applied **data augmentation transformations** (e.g., flipping, color jitter, cropping) more aggressively on **minority classes** to synthetically increase their sample size and variability.

---

## 4. Class Distribution

The following table shows the number of images for each class in the dataset:

| Class | Label     | Images |
|-------|-----------|--------|
| 0     | Class 0   | 4911   |
| 1     | Class 1   | 2662   |
| 2     | Class 2   | 2093   |
| 3     | Class 3   | 2625   |
| 4     | Class 4   | 3086   |
| 5     | Class 5   | 7377   |
| 6     | Class 6   | 2016   |
| 7     | Class 7   | 1415   |
| 8     | Class 8   | 3933   |
| 9     | Class 9   | 2955   |
| 10    | Class 10  | 14432  |

---

## 5. Evaluation Results

The model was evaluated on a separate test set. The following table summarizes the overall performance:

| Metric        | Score   |
|---------------|---------|
| Accuracy      | 81.37%  |
| Macro F1      | 0.7550  |
| Micro F1      | 0.8137  |
| Weighted F1   | 0.8146  |

### **Per-Class Accuracy**

| Class | Accuracy |
|-------|----------|
| 0     | 75.46%   |
| 1     | 62.28%   |
| 2     | 73.53%   |
| 3     | 67.01%   |
| 4     | 72.16%   |
| 5     | 83.83%   |
| 6     | 52.28%   |
| 7     | 91.66%   |
| 8     | 78.18%   |
| 9     | 86.02%   |
| 10    | 94.32%   |

---

### **Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.7294    | 0.7546 | 0.7418   | 4911    |
| 1     | 0.6826    | 0.6228 | 0.6513   | 2662    |
| 2     | 0.7413    | 0.7353 | 0.7383   | 2093    |
| 3     | 0.6484    | 0.6701 | 0.6590   | 2625    |
| 4     | 0.7920    | 0.7216 | 0.7552   | 3086    |
| 5     | 0.8340    | 0.8383 | 0.8361   | 7377    |
| 6     | 0.4561    | 0.5228 | 0.4872   | 2016    |
| 7     | 0.7818    | 0.9166 | 0.8439   | 1415    |
| 8     | 0.8554    | 0.7818 | 0.8170   | 3933    |
| 9     | 0.7919    | 0.8602 | 0.8247   | 2955    |
| 10    | 0.9583    | 0.9432 | 0.9507   | 14432   |
| **Accuracy** |       |        | **0.8137** | **47505** |
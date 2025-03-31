# Classifier Model for Garment Classification

## 1. Introduction
The goal of this project was to train a deep learning model to classify clothing items into categories such as **Upper Garment**, **Lower Garment**, and **Both** types. We aimed to structure classifications hierarchically by **gender** (e.g., men, women), and **type** (e.g., shirt, pants).

To achieve this, we utilized a pre-trained deep learning model and adapted it for improved classification performance.

---

## 2. Model Selection and Adaptation

This model utilizes a **ResNet-50** backbone for feature extraction, leveraging the pretrained weights from ImageNet to enable strong feature representation from clothing images.

### **Model Architecture**
The `ClothingClassifier` is built as follows:

- **Backbone**: A pre-trained **ResNet-50** is used, with its final classification layer removed (`resnet.children()[:-1]`). The output feature map is globally averaged and passed to the next stage.
- **Fully Connected Layer**: A linear layer projects the ResNet features into a 1024-dimensional space.
- **Classification Head**: A deep, fully connected head processes the 1024-dimensional feature vector through multiple layers with batch normalization, ReLU activations, and dropout regularization. The final output is a linear layer projecting to `num_classes`.

The classifier head consists of:
- Linear(1024 → 512) → BatchNorm1d → ReLU → Dropout(0.2)
- Linear(512 → 256) → BatchNorm1d → ReLU → Dropout(0.2)
- Linear(256 → 128) → BatchNorm1d → ReLU → Dropout(0.2)
- Linear(128 → `num_classes`)

### **Transfer Learning and Fine-Tuning**
To balance learning new task-specific features while preserving the pretrained knowledge from ResNet-50, the model includes an option to **freeze early layers** of the backbone. The number of frozen layers can be adjusted using the `num_frozen_resnet_layers` argument during initialization.

---

## 3. Class Distribution

The following table shows the number of images for each class in the dataset:

| Class | Label     | Images |
|-------|-----------|--------|
| 0     | Class 0 (Blouse)      | 24557 |
| 1     | Class 1 (Cardigan)    | 13311 |
| 2     | Class 2 (Jacket)      | 10467 |
| 3     | Class 3 (Sweater)     | 13123 |
| 4     | Class 4 (Tank)        | 15429 |
| 5     | Class 5 (Tee)         | 36887 |
| 6     | Class 6 (Top)         | 10078 |
| 7     | Class 7 (Jeans)       | 7076  |
| 8     | Class 8 (Shorts)      | 19666 |
| 9     | Class 9 (Skirts)      | 14773 |
| 10    | Class 10 (Dress)      | 72158 |

---

## 4. Training Strategy

### **Dataset Split**
We used a stratified split to maintain class distribution across all subsets:
- **Training set:** 60%
- **Validation set:** 20%
- **Test set:** 20%

### **Hyperparameter Tuning**
We performed **hyperparameter tuning** to determine the optimal number of frozen layers in the ResNet-50 backbone. This helped balance between retaining pretrained knowledge and adapting to the new task, improving convergence and generalization.

### **Handling Class Imbalance**
To address class imbalance:
- We used a **weighted cross-entropy loss function**, giving more weight to minority classes to prevent the model from overfitting to dominant categories.
- We applied **data augmentation transformations** (e.g., flipping, color jitter, cropping) more aggressively on **minority classes** to synthetically increase their sample size and variability.

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
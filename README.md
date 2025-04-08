# CNN-Showdown-ResNet-vs-VGG

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white)

This project provides an in-depth comparison of two seminal Convolutional Neural Network (CNN) architectures, **VGG-16** and **ResNet-18**, for a multi-class image classification task. The entire pipeline, from data analysis to model training and evaluation, is implemented using PyTorch to determine which architecture provides better performance and training stability on a balanced, real-world dataset.

---

## 📊 Dataset Overview

The project utilizes a dataset of 30,000 images, equally distributed across three real-world categories: **dogs, food, and vehicles**. Each class contains exactly 10,000 images, creating a perfectly balanced dataset ideal for training a classification model without class-imbalance bias.

---

## 🧠 Model Architectures

Two distinct CNN architectures were implemented and compared.

### 1. VGG-16 (Version C)
The VGG-16 model is characterized by its simplicity and depth, using small 3x3 convolutional filters stacked in sequence. The core idea is to increase network depth to learn a richer hierarchy of features. This implementation includes **Batch Normalization** after each convolutional layer and **Dropout** in the classifier for regularization, which are essential for stabilizing the training of such a deep network.

### 2. ResNet-18
The ResNet-18 model introduces **residual (shortcut) connections** to address the vanishing gradient problem in deep networks. Instead of learning a direct mapping, layers learn a residual mapping, which allows for the training of much deeper models without performance degradation. The architecture is built from a series of `ResidualBlock` modules that contain the shortcut connections.

---

## ⚙️ Training & Optimization

Both models were trained using a comprehensive strategy to ensure a fair comparison and achieve optimal performance:
* **Data Augmentation**: To improve generalization and prevent overfitting, the training data was augmented with random horizontal flips, rotations, and color jitter.
* **L2 Regularization**: Weight decay was added to the optimizer to reduce overfitting.
* **Learning Rate Scheduling**: A `StepLR` scheduler was used to decrease the learning rate at later stages of training for finer adjustments.

---

## 🏆 Performance & Evaluation

The final evaluation on the held-out test set clearly demonstrates the superiority of the ResNet-18 architecture for this task.

![Optimizer Performance Comparison](assets/f8.jpg)

* **Final Test Accuracy**:
    * **VGG-16**: 89.71%
    * **ResNet-18**: **94.22%**

The ResNet-18 model not only achieved a significantly higher final accuracy but also demonstrated faster and more stable convergence during training. The residual connections in ResNet were crucial in enabling the model to effectively learn complex patterns from the data without being hindered by the vanishing gradient problem, a common issue in deeper networks like VGG.

---

## 🚀 How to Run

1.  **Prerequisites**: Ensure you have Python and the necessary libraries installed from `requirements.txt`.
2.  **Dataset**: Download the dataset and place the unzipped folder in the project's root directory.
3.  **Execution**: Run the Jupyter Notebook `CNN-Showdown-ResNet-vs-VGG.ipynb` sequentially. The notebook handles data preparation, model training, and evaluation.## Final Results

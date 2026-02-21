📌 CIFAR-100 Fine-Grained Image Classification
📖 Project Overview

This project implements fine-grained image classification on the CIFAR-100 dataset using advanced Deep Learning techniques.

The objective is to design, train, and evaluate:

A Custom Convolutional Neural Network (CNN)

A Transfer Learning model (MobileNetV2 pre-trained on ImageNet)

The project follows a complete deep learning pipeline including preprocessing, data augmentation, hyperparameter tuning, learning rate scheduling, fine-tuning, and detailed performance analysis.

📂 Dataset

This project uses the CIFAR-100 dataset.

60,000 color images (32×32 resolution)

100 fine-grained classes

500 training images per class

100 test images per class

Classes grouped into 20 superclasses

The dataset is loaded directly from TensorFlow/Keras.

Example images:

<p align="center"> <img src="https://www.cs.toronto.edu/~kriz/cifar-100-sample.png" width="500"> </p>
🧠 Models Implemented
🔹 1. Custom CNN

Built from scratch using:

Multiple Conv2D layers

Batch Normalization

ReLU activations

MaxPooling

Dropout (regularization)

Global Average Pooling

Dense classification head (100 classes)

Loss Function:

Categorical Crossentropy

Metrics:

Accuracy

Top-5 Accuracy

Optimizer:

Adam

Learning Rate Scheduler:

ReduceLROnPlateau

🔹 2. Transfer Learning Model

Backbone:

MobileNetV2 (Pre-trained on ImageNet)

Training Strategy:

Stage 1 – Feature Extraction

Freeze backbone

Train custom classification layers

Stage 2 – Fine-Tuning

Unfreeze backbone

Train entire model with low learning rate

Advantages:

Faster convergence

Better feature extraction

Improved generalization

⚙️ Data Preprocessing

Normalization (pixel values scaled to [0,1])

One-hot encoding of labels

Train-validation split (80-20)

Data Augmentation:

Random Horizontal Flip

Random Rotation

Random Zoom

Random Contrast

Data augmentation improves generalization and reduces overfitting due to limited training samples per class.

📊 Evaluation Metrics

Test Accuracy

Top-5 Accuracy

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

📈 Results Summary
Model	Test Accuracy	Top-5 Accuracy
Custom CNN	~XX%	~XX%
Transfer Learning	~XX%	~XX%

(Replace XX% with your actual results after training)

Transfer Learning outperformed the Custom CNN due to pre-trained feature representations learned from ImageNet.

📁 Project Structure
cifar100-fine-grained-image-classification/
│
├── notebook.ipynb
├── best_cifar100_model.h5
├── requirements.txt
├── README.md
🛠️ Requirements

Python 3.8+

TensorFlow

NumPy

Matplotlib

Seaborn

Scikit-learn

Install dependencies:

pip install -r requirements.txt
🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/cifar100-fine-grained-image-classification.git

Navigate to the folder:

cd cifar100-fine-grained-image-classification

Open the notebook:

jupyter notebook

Run all cells.

🔍 Key Learnings

Importance of Batch Normalization for stable training

Dropout reduces overfitting

Data augmentation improves generalization

Transfer learning significantly boosts performance

Learning rate scheduling helps optimization

Fine-grained classification is challenging due to subtle class differences

🚧 Challenges

Small image resolution (32×32)

100 visually similar classes

Risk of overfitting

Large confusion matrix (100×100)

🔮 Future Improvements

Try EfficientNet or ResNet50

Use stronger augmentation (CutMix, MixUp)

Implement Vision Transformers

Train longer with cosine annealing scheduler

Ensemble multiple models

👤 Author

Tirth

Fine-Grained Image Classification Project
Deep Learning Coursework

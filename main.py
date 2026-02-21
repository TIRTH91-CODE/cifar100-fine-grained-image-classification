
# 1. IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# 2. LOAD DATASET

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

num_classes = 100

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)


# 3. VISUALIZE SAMPLE IMAGES

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[i])
    plt.axis("off")
plt.suptitle("Sample CIFAR-100 Images")
plt.show()

# 4. PREPROCESSING


# Normalize (0–1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Train-validation split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

print("Training:", x_train.shape)
print("Validation:", x_val.shape)


# 5. DATA AUGMENTATION

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


# 6. CUSTOM CNN MODEL

def build_custom_cnn():
    model = keras.Sequential([
        data_augmentation,

        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

custom_model = build_custom_cnn()

custom_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5)]
)

custom_model.summary()


# 7. LEARNING RATE SCHEDULER

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# 8. TRAIN CUSTOM CNN

history_custom = custom_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[lr_scheduler]
)


# 9. PLOT TRAINING CURVES

def plot_history(history, title):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Val")
    plt.title(title + " Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Val")
    plt.title(title + " Loss")
    plt.legend()

    plt.show()

plot_history(history_custom, "Custom CNN")

 10. TRANSFER LEARNING (MobileNetV2)

base_model = keras.applications.MobileNetV2(
    input_shape=(32,32,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze backbone

inputs = keras.Input(shape=(32,32,3))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

transfer_model = keras.Model(inputs, outputs)

transfer_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5)]
)


# 11. STAGE 1 TRAINING
history_stage1 = transfer_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64
)

 12. STAGE 2 FINE-TUNING

base_model.trainable = True

transfer_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5)]
)

history_stage2 = transfer_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64
)


# 13. FINAL EVALUATION

print("Custom CNN Test Results:")
custom_model.evaluate(x_test, y_test)

print("\nTransfer Learning Test Results:")
transfer_model.evaluate(x_test, y_test)


# 14. CLASSIFICATION REPORT

y_pred = np.argmax(transfer_model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))


# 15. CONFUSION MATRIX

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# 16. SAVE MODEL

transfer_model.save("best_cifar100_model.h5")

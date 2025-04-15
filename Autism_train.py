import tensorflow as tf
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set Dataset Path
data_dir = r'C:\Users\YourUsername\AutismDataset\data'  # Update this path

# Load Dataset
data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='binary',  # Binary classification (Autistic vs Non_Autistic)
    image_size=(256, 256),
    batch_size=32,
    shuffle=True
)

# Print class names
print(f"Classes: {data.class_names}")

# Normalize Data
data = data.map(lambda x, y: (x / 255, y))

# Split Data (Train, Validation, Test)
train_size = int(len(data) * 0.75) + 1
val_size = int(len(data) * 0.15)
test_size = int(len(data) * 0.10)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build CNN Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),  # Regularization
    BatchNormalization(),

    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Setup Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train Model
history = model.fit(
    train,
    epochs=50,
    validation_data=val,
    callbacks=[early_stop, reduce_lr]
)

# Plot Performance
plt.figure()
plt.plot(history.history['loss'], color='red', label='Loss')
plt.plot(history.history['val_loss'], color='orange', label='Validation Loss')
plt.title('Loss Over Time')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='Accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='Validation Accuracy')
plt.title('Accuracy Over Time')
plt.legend()
plt.show()

# Evaluate Model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    z = model.predict(X)
    pre.update_state(y, z)
    re.update_state(y, z)
    acc.update_state(y, z)

print(f'Precision: {pre.result().numpy()}')
print(f'Recall: {re.result().numpy()}')
print(f'Accuracy: {acc.result().numpy()}')

# Test Model on a New Image
test_img_path = r'C:\Users\YourUsername\AutismDataset\test\Autistic.134.jpg'  # Update this path
img = cv2.imread(test_img_path)
resize = tf.image.resize(img, (256, 256))

plt.imshow(resize.numpy().astype(int))
plt.show()

z = model.predict(np.expand_dims(resize/255, 0))

if z > 0.5:
    print('Predicted: Not Autistic')
else:
    print('Predicted: Autistic')

# Save Model
model.save('autism_detection_model.h5')
print("Model saved successfully!")

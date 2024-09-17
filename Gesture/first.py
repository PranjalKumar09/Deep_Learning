import matplotlib.pyplot as plt
import tensorflow as tf

# Recreate your model architecture here (this is just an example; replace with your actual model)
model = tf.keras.Sequential([
    # Example layers - replace these with the exact architecture used for training
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Adjust this to your actual number of classes
])

# Load the weights into the model
model.load_weights('model_epoch_10.weights.h5', by_name=True, skip_mismatch=True)

# Optionally, compile the model (depending on what you plan to do next, like inference or further training)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now you can use the model for predictions or further training


# Plot the training history (assuming you have access to the previous history)
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)  # Adjust range as per your history

    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# Visualizing confusion matrix (Optional)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Assuming test_generator is defined and already used for validation
y_true = test_generator.classes  # True labels
y_pred = model.predict(test_generator)  # Model predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Call the function to plot training history (assuming `history` is your training history)
plot_training_history(history)

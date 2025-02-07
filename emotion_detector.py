import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

class EmotionDetector:
    def __init__(self, model_path=None):
        # Load the model if a path is provided, otherwise build the model
        self.model = self.build_model() if model_path is None else tf.keras.models.load_model(model_path)

    def build_model(self):
        # Define the model architecture
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # Use (48, 48, 1) for grayscale
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # Output layer for 7 emotions
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, test_dir, epochs=10, batch_size=64):
        # Data augmentation for training and validation
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Data generators for training and validation
        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(48, 48), color_mode="grayscale", batch_size=batch_size, class_mode='categorical'
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=(48, 48), color_mode="grayscale", batch_size=batch_size, class_mode='categorical'
        )

        # Training the model
        history = self.model.fit(train_generator, validation_data=test_generator, epochs=epochs)
        self.model.save('models/emotion_model.h5')  # Save the trained model
        return history

    def predict_emotion(self, img):
        # Resize image to (48, 48) and convert to grayscale if necessary
        img = cv2.resize(img, (48, 48))

        # Check if the image is grayscale or RGB and reshape accordingly
        if len(img.shape) == 2:  # Grayscale image (48, 48)
            img = np.expand_dims(img, axis=-1)  # Reshape to (48, 48, 1)
        
        img = img.astype('float32') / 255.0  # Normalize the image
        
        # Add batch dimension: (1, 48, 48, 1) for grayscale
        img = np.expand_dims(img, axis=0)  # Shape should be (1, 48, 48, 1)
        
        # Predict the emotion
        predictions = self.model.predict(img)
        emotion_index = predictions.argmax()  # Get the index of the highest predicted emotion
        return emotion_index

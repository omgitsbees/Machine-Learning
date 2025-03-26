import cv2 
import numpy as np 
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.applications.vgg16 import preprocess_input 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.optimizers import Adam 
from sklearn.preprcossing import LabelEncoder 
from sklearn.svm import SVC 
import dlib 
from tensorflow.keras.models import load_model 
import coremltools 
import tensorflow as tf 

# Data Collection 
def load_images_from_folder(folder):
    images = [] 
    for filename in os.listdir(folder): 
        img = cv2.imread(os.path.join(folder, filename)) 
        if img is not None:
            images.append(img)
    return images 

# Data Preprocessing 
def preprocess_images(images):
    processed_images = [] 
    for img in images:
        img = cv2.resize(img, (224, 224)) # Reize to 224x224
        img = preprocess_input(img) # Preprocess for VGG16 
        processed_images.append(img)
    return np.array(processed_images)

# Load pre-trained VGG16 model + higher level layers 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the base model
x = base_model.output 
x = Flatten()(x) 
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x) # Assuming 10 classes 

# Define the model 
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model 
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 

# Face Detection
def detect_faces(image):
    face_cascase = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces 

# Face Regnition 
def train_face_recognition_model(embeddings, labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(embeddings, labels)
    return recognizer, le

# Facial Landmark Detection 
def detect_facial_landmarks(image, faces):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    landmarks = []
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(image, rect) 
        landmarks.append([(p.x, p.y) for p in shape.parts()])
    return landmarks 

# Load pre-trained emotion detection model 
emotion_model = load_model('path_to_emotion_model.h5')

# Emotion Detection
def detect_emotions(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float') / 255
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    emotion_prediction = emotion_model.predict(face)
    return emotion_prediction 

# 3D Face Reconstruction using PRNet 
def reconstruct_3d_face(image):
    # Load PRNet model
    prnet_model = load_model('path_to_prenet_model.h5')
    
    # Preprocess image 
    img = cv2.resize(image, (256, 256)) 
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    
    # Predict 3d face 
    pos_map = prnet_model.predict(img)[0] 
    
    # Post-process to get 3D coordinates 
    vertices = np.zeros((256, 256, 3))
    for i in range(256):
        for i in range[256]:
            z = pos_map[i, j, 2] * 255
            vertices[i, j] = [i, j, z]
            
    return vertices 

# Face Anti-Spoofing using a simple CNN
def build_anti_spoofing_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model 

# Train the anti-spoofing model
def train_anti_spoofing_model(model, train_data, train_labels):
    model.fit(train_data, train_labels, epochs=10, batch_size=32)
    
# Detect spoofing
def detect_spoofing(image, model):
    img = cv2.rezize(image, (64, 64))
    img = img.astype('float32') / 255 
    img = np.expand_dims(img, axis=0) 
    prediction = model.predict(img)
    return prediction > 0.5 

# Convert the Keras model to Core ML 
coreml_model = coremltools.converters.convert(model, input_names=['image'], output_names=['output'])

# Save the Core ML model
coreml_model.save('ImageRecognitionModel.mlmodel')

# Example usage
folder_path = 'path_to_folder'
images = load_images_from_folder(folder_path)
processed_images = preprocess_images(images)

# Assuming you have labels and processed_images
# model.fit(processed_images, labels, epochs=10, batch_size=32)

# Example usage for face detection
# faces = detect_faces(image)

# Example usage for face recognition
# recognizer, le = train_face_recognition_model(embeddings, labels)

# Example usage for facial landmark detection
# landmarks = detect_facial_landmarks(image, faces)

# Example usage for emotion detection
# emotion = detect_emotions(face)

# Example usage for 3D face reconstruction
# reconstructed_face = reconstruct_3d_face(image)

# Example usage for face anti-spoofing
# anti_spoofing_model = build_anti_spoofing_model()
# train_anti_spoofing_model(anti_spoofing_model, train_data, train_labels)
# is_spoof = detect_spoofing(image, anti_spoofing_model)
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report

# 1. Prepare your dataset (same as before for simplicity)
data = [
    ("This video uses copyrighted music without permission.", "copyright_infringement"),
    ("The content in this video is original and created by me.", "not_infringement"),
    ("This is a fair use commentary on another video.", "not_infringement"),
    ("The visuals in this video are directly copied from another source.", "copyright_infringement"),
    ("I have obtained the necessary licenses for all media used.", "not_infringement"),
    ("This short clip falls under the public domain.", "not_infringement"),
    ("The majority of this video is someone else's work.", "copyright_infringement"),
    ("This is a parody and transformative work.", "not_infringement"),
    ("The background music is a royalty-free track from a known source.", "not_infringement"),
    ("This video contains substantial portions of a copyrighted movie.", "copyright_infringement"),
    ("My video is a reaction and critique, falling under fair dealing.", "not_infringement"),
    ("I have explicit permission from the copyright holder to use this material.", "not_infringement"),
    ("This is a blatant unauthorized reproduction.", "copyright_infringement"),
    ("The similarities are purely coincidental.", "not_infringement"),
]

texts = [item[0] for item in data]
labels = [item[1] for item in data]
label_map = {"not_infringement": 0, "copyright_infringement": 1}
numerical_labels = np.array([label_map[label] for label in labels])

# 2. Tokenization and Sequencing
# Convert text to sequences of integers
max_words = 10000  # Maximum number of words to keep in the vocabulary
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>") # <unk> for out-of-vocabulary words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to have the same length
max_len = 100  # Maximum length of a sequence
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 3. Load GloVe Embeddings
embedding_dim = 100  # Dimension of the GloVe embeddings
glove_file_path = 'glove.6B.100d.txt'  # Replace with the actual path to your GloVe file
embeddings_index = {}
try:
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
except FileNotFoundError:
    print(f"Error: GloVe embeddings file not found at {glove_file_path}")
    exit()

# Create an embedding matrix
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, numerical_labels, test_size=0.2, random_state=42)

# 5. Build the LSTM Model
model = Sequential([
    Embedding(num_words, embedding_dim, embeddings_initializer='constant', input_length=max_len, trainable=False),
    LSTM(64, return_sequences=False), # Single LSTM layer
    Dropout(0.5),
    Dense(1, activation='sigmoid') # Binary classification output
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 6. Train the Model
epochs = 10
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# 7. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

predictions = model.predict(X_test)
binary_predictions = np.round(predictions).flatten().astype(int)

print("\nClassification Report:")
print(classification_report(y_test, binary_predictions, target_names=label_map.keys()))

# 8. Example of predicting on new, unseen text
new_texts = [
    "This video features a cover song performed by me.",
    "The background music is a royalty-free track.",
    "This content directly re-uploads someone else's entire video.",
    "This is a compilation of short clips under fair use guidelines.",
    "The audio track is identical to a copyrighted song.",
]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_len, padding='post', truncating='post')
new_predictions = model.predict(new_padded_sequences)
new_binary_predictions = np.round(new_predictions).flatten().astype(int)

print("\nPredictions on new texts:")
for text, prediction in zip(new_texts, new_binary_predictions):
    predicted_label = "copyright_infringement" if prediction == 1 else "not_infringement"
    print(f"Text: '{text}' - Predicted Label: {predicted_label}")
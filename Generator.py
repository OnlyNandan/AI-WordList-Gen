import string
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def generate_dataset(num_samples, max_length):
    characters = string.ascii_letters + string.digits + string.punctuation
    passwords = [''.join(random.choice(characters) for _ in range(max_length)) for _ in range(num_samples)]
    labels = [1 if 'password' in password.lower() else 0 for password in passwords]
    return passwords, labels

def generate_ai_password(model, char_to_int, int_to_char, max_length, temperature=1.0):
    password = ''
    x_input = np.zeros((1, max_length, 1))

    for _ in range(max_length):
        x_input[0, _, 0] = random.randint(0, len(char_to_int)-1)

    for _ in range(max_length):
        prediction = model.predict(x_input, verbose=0)[0][0]
        
        # Apply temperature to control randomness
        prediction = np.log(prediction) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)

        index = np.argmax(np.random.multinomial(1, prediction, 1))
        char = int_to_char[index]
        password += char

        # Update input sequence for the next character
        x_input = np.roll(x_input, -1, axis=1)
        x_input[0, -1, 0] = index

    return password

# Generate a dataset
num_samples = 10000
max_length = 16
passwords, labels = generate_dataset(num_samples, max_length)

# Create character mappings
characters = sorted(list(set(''.join(passwords))))
char_to_int = {char: i for i, char in enumerate(characters)}
int_to_char = {i: char for i, char in enumerate(characters)}

# Convert passwords to numerical sequences
X = np.array([[char_to_int[char] for char in password] for password in passwords])
y = np.array(labels)

# Reshape X for LSTM input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Generate passwords using the trained model with different temperatures
generated_password_cool = generate_ai_password(model, char_to_int, int_to_char, max_length, temperature=0.5)
generated_password_normal = generate_ai_password(model, char_to_int, int_to_char, max_length, temperature=1.0)
generated_password_hot = generate_ai_password(model, char_to_int, int_to_char, max_length, temperature=1.5)

print("Generated Password (Cool):", generated_password_cool)
print("Generated Password (Normal):", generated_password_normal)
print("Generated Password (Hot):", generated_password_hot)

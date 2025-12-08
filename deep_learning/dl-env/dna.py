import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

print("he3lo0 world")
data = pd.read_csv("./all_classifcation_and_seqs_aln.csv")

data = data.dropna()
#encode each species to a specific number
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

#correspond letter to number
letter_mapping = {
'A':0,
'G':2,
'T':1,
'C':3,
'-':4
}
#goes through each row then goes through each letter in sequence encode DNA sequence
encoded_char = []
encoded_column = []
for text_string in data['sequence']:
    each_encoded_char= []
    for char in text_string:
        each_encoded_char.append(letter_mapping[char])
    encoded_char.append(each_encoded_char)
data['new_sequence'] = encoded_char

print(data.head())

#tells me the dimensions of the sequence
print(data['sequence'].str.len())


#using x-value:DNA sequence to predict y-value: speicies
X = np.array(encoded_char)
y = data["species"]

# TODO : Split the data into testing and training data. Use a 20% split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)
# run this to see if you implemented the above block correctly
assert math.isclose(len(X_train), .8*len(data), rel_tol=1), f"\033[91mExpected {.8*len(data)} but got {len(X_train)}\033[0m"
assert math.isclose(len(X_test), .2*len(data), rel_tol=1), f"\033[91mExpected {.2*len(data)} but got {len(X_test)}\033[0m"


# TODO create a neural network with tensorflow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1267,input_shape = [27040]),
    #tf.keras.layers.Dense(67, activation= 'sigmoid'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Softmax()
])

# TODO set your learning rate
lr = 0.00005

#TODO Compile your model with a selected optimizer and loss function
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate= lr),
    metrics = ['accuracy'])

# TODO: fit your model with X_train and Y_train
history = model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs = 200)



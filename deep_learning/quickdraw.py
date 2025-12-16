import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


# Helper function to show images from your dataset
def show_image(data):
    index = random.randint(0, len(data)-1)
    img2d = data[index].reshape(28, 28)
    plt.figure()
    plt.imshow(img2d)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # Load your first cateogry of images
data_pencil = np.load("pencil.npy")

# Now use the function above to show an image from your data
show_image(data_pencil)
# Now load your second category of images
data_knife = np.load("knife.npy")
# visualize your second category of images
show_image(data_knife)
# Now load your third category of images
data_fork = np.load("fork.npy")
# visualize your third category of images
show_image(data_fork)
# Now load your 4th category of images
data_asparagus = np.load("asparagus.npy")
# visualize your 4th category of images
show_image(data_asparagus)
# Now load your 5th category of images
data_crayon = np.load("crayon.npy")
# visualize your 5th category of images
show_image(data_crayon)
# define X by combining all loaded data from above
X = np.vstack([data_crayon,data_pencil,data_knife,data_asparagus,data_fork])
# verify the X was defined correctly
assert X.shape[1] == 784
assert X.shape[0] >= 550000
# define y by creating an array of labels that match X
y = np.concatenate([
    np.full(len(data_asparagus), 0, dtype= object),
    np.full(len(data_knife), 1, dtype= object),
    np.full(len(data_crayon), 2, dtype= object),
    np.full(len(data_fork), 3, dtype= object),
    np.full(len(data_pencil), 4, dtype= object),
])

y = y.astype(int)
# verify that y is the same length as X
assert len(y) == len(X)
# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# Define your model with the correct input shape and appropriate layers
# TODO create a neural network with tensorflow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1267,input_shape = [784]),
    #tf.keras.layers.Dense(67, activation= 'sigmoid'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Dense(677, activation = 'relu'),
    tf.keras.layers.Softmax()
])

# Compile your model
# TODO set your learning rate
lr = 0.00005

#TODO Compile your model with a selected optimizer and loss function
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate= lr),
    metrics = ['accuracy'])

# TODO: fit your model with X_train and Y_train
history = model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs = 100)

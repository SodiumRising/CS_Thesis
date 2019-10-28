import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
#from tqdm import tqdm_notebook
import os
#%matplotlib inline

IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
TEST_SIZE = 0.1
RANDOM_STATE = 2018
# Model
NO_EPOCHS = 6
BATCH_SIZE = 128

train_images = np.load('C:\Kazushiji/kmnist-train-imgs.npz')['arr_0']
test_images = np.load('C:\Kazushiji/kmnist-test-imgs.npz')['arr_0']
train_labels = np.load('C:\Kazushiji/kmnist-train-labels.npz')['arr_0']
test_labels = np.load('C:\Kazushiji/kmnist-test-labels.npz')['arr_0']
char_df = pd.read_csv('C:\Kazushiji/kmnist_classmap.csv', encoding='utf-8')

def main():

    print("KMNIST train shape:", train_images.shape)
    print("KMNIST test shape:", test_images.shape)
    print("KMNIST train shape:", train_labels.shape)
    print("KMNIST test shape:", test_labels.shape)
    print("KMNIST character map shape:", char_df.shape)
    print('Percent for each category:', np.bincount(train_labels)/len(train_labels)*100)

    X, y = data_preprocessing(train_images, train_labels)
    X_test, y_test = data_preprocessing(test_images, test_labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("KMNIST train -  rows:", X_train.shape[0], " columns:", X_train.shape[1:4])
    print("KMNIST valid -  rows:", X_val.shape[0], " columns:", X_val.shape[1:4])
    print("KMNIST test -  rows:", X_test.shape[0], " columns:", X_test.shape[1:4])

    # plot_count_per_class(np.argmax(y_train, axis=1))
    # get_count_per_class(np.argmax(y_train, axis=1))

    # plot_count_per_class(np.argmax(y_val, axis=1))
    # get_count_per_class(np.argmax(y_val, axis=1))

    makeCNNModel(X_train, y_train, X_val, y_val)


# data preprocessing
def data_preprocessing(images, labels):

    out_y = keras.utils.to_categorical(labels, NUM_CLASSES)
    num_images = images.shape[0]
    x_shaped_array = images.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


def plot_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    g = sns.countplot(ydf[0], order=np.arange(0, 10))
    g.set_title("Number of items for each class")
    g.set_xlabel("Category")

    plt.show()


def get_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    # Get the count for each label
    label_counts = ydf[0].value_counts()

    # Get total number of samples
    total_samples = len(yd)

    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        label_char = char_df[char_df['index'] == label]['char'].item()
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{}({}):   {} or {}%".format(label, label_char, count, percent))

def makeCNNModel(X_train, y_train, X_val, y_val):

    # Model
    model = Sequential()

    # Add convolution 2D
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same",
                     kernel_initializer='he_normal', input_shape=(IMG_ROWS, IMG_COLS, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Add dropouts to the model
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))

    # Add dropouts to the model
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Add dropouts to the model
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Compile the model
    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NO_EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val))

    model.save('the_best_around.model')


main()

# https://www.kaggle.com/gpreda/classifying-cursive-hiragana-kmnist-using-cnn

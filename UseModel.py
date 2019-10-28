import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
import os
import csv


def main():

    IMG_ROWS = 28
    IMG_COLS = 28

    new_model = tf.keras.models.load_model('the_best_around.model')
    characters = getHandwriting()
    num_images = characters.shape[0]
    x_shaped_array = characters.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped_array / 255
    makePrediction(new_model, out_x)

def getHandwriting():

    # https://stackoverflow.com/questions/40727793/how-to-convert-a-grayscale-image-into-a-list-of-pixel-values

    img = Image.open(r'C:\Users\loofa\Desktop\TestHiragana\wo.png').convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

    # At this point the image's pixels are all in memory and can be accessed
    # individually using data[row][col].

    # For example:
    for row in data:
        for value in row:
            print(str(255 - value) + " ", end="")
            # print(' '.join('{:3}'.format(255-value) for value in row))

    return data

def makePrediction(new_model, characters):

    CATEGORIES = ["O", "KI", "SU", "TSU", "NA", "HA", "MA", "YA", "RE", "WO"]

    predictions = new_model.predict(characters)
    print(CATEGORIES[int(predictions[0][0])])


main()

#  https://stackoverflow.com/questions/49070242/converting-images-to-csv-file-in-python

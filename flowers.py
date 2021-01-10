import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import tkinter as tk

window = tk.Tk()

batch_size = 32
img_height = 180
img_width = 180
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

flower_file = sys.argv[1]
print(flower_file)

new_model = tf.keras.models.load_model('modelsave')

# Check its architecture
# new_model.summary()

'''
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
'''

dir_path = os.path.dirname(os.path.realpath(__file__))
rose_path = os.path.join(dir_path, flower_file)

img = keras.preprocessing.image.load_img(
    rose_path, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = new_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

outcome = tk.Label(
    text="This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)),
    foreground= "pink",
    background= "blue",
    width=100,
    height=50

)

outcome.pack()

print(
    "CLONED TRAIN: This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


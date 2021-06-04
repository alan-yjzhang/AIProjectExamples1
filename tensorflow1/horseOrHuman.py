import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# !wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
# https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip

train_dir = "/tmp/horse-or-human"
train_datagen = ImageDataGenerator(rescale= 1.0/255)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(300,300), batch_size=128, class_mode='binary'
)
validation_dir = "/tmp/validation-horse-or-human"
test_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(300,300), batch_size=128, class_mode='binary'
)

train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
train_human_dir = os.path.join('/tmp/horse-or-human/humans')
train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
print(train_horse_names[:10], train_human_names[:10])
print('total horse images:', len(os.listdir(train_horse_dir)))
print('total human images:', len(os.listdir(train_human_dir)))

model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3), activation='relu',input_shape=(300,300,1)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.summary()


from tensorflow.keras.optimizers import  RMSprop
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=15, verbose=1)

# history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=15,
#                     validation_data=validation_generator,
#                     validation_steps=8, verbose=2)


# Visualize the intermediate results
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array, load_img
# Let's define a new Model that will take an image as input, and will output intermediate representations for all layers in the previous odel after the first
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]

img_path = random.choice(horse_img_files + human_img_files)
img = load_img(img_path, target_size=(300,300)) # this is a PIL image
x = img_to_array(img) # Numpy array with shape (150,150,3)
x = x.reshape((1,)+x.shape) # Numpy array with shape (1,150,150,3)
x /= 255.0

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv /maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1] # number of features in the feature map
    # the feature map has shape [1, size, size, n_features]
    size = feature_map.shape[1]
    display_grid = np.zeros((size,size*n_features))
    for i in range(n_features):
      # post processing the feature to make it visually palatable
      x = feature_map[0,:,:,i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i*size : (i+1)*size ]  = x
    # Display the grid
    scale = 20.0 /n_features
    plt.figure(figsize=(scale*n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

############################################################
# Evaluation
import  numpy as np
from keras_preprocessing import image
# from google.colab import files
# uploaded = files.upload()

# for fn in uploaded.keys():
#
#     #predicting images
#     path = '/content/' + fn
#     img = image.load_img(path, target_size=(300,300))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(classes[0])
#     if classes[0] > 0.5:
#         print(fn + " is a human")
#     else:
#         print(fn + " is a horse")



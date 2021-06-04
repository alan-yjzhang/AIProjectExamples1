import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])

train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images /255.0

test_images = test_images.reshape(10000,28,28,1)
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(64,(3,3), activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images, test_labels)

# Visualizing the convolution and pooling
from tensorflow.keras import models
f, axarr = plt.subplots(3,4)
firstImage = 2
secondImage = 3
thirdImage = 5
convNumber = 4

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
for x in range(0,4):
    f1 = activation_model.predict(test_images[firstImage].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0,:,:,convNumber], cmap="inferno")
    axarr[0,x].grid(False)
    f2 = activation_model.predict(test_images[secondImage].reshape(1,28,28,1))[x]
    axarr[1,x].imshow(f2[0,:,:,convNumber], cmap="inferno")
    axarr[1,x].grid(False)
    f3 = activation_model.predict(test_images[thirdImage].reshape(1,28,28,1))[x]
    axarr[2,x].imshow(f3[0,:,:,convNumber], cmap="inferno")
    axarr[2,x].grid(False)





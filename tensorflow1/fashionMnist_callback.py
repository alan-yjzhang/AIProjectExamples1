import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])

train_images = train_images /255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.4):
            print("\Loss is low so cancelling training!")
            self.model.stop_training = True

callback = myCallback()
model.fit(train_images, train_labels, epochs=5, callbacks=[callback])

model.evaluate(test_images, test_labels)








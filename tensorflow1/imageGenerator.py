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




import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, Resizing
from tensorflow.keras.models import Sequential

from datetime import datetime

IMG_SIZE=64

def get_conv(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    '../data/selfmade',
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32)

model = get_conv()
history = model.fit(train_ds, validation_data=test_ds, epochs=5)
model.save(f'../models/{datetime.now().strftime("%H%M")}.keras')

train_images, train_labels = next(iter(train_ds.unbatch().batch(len(train_ds))))
test_images, test_labels = next(iter(test_ds.unbatch().batch(len(test_ds))))

for type, images, labels in [['train', train_images, train_labels], ['test', test_images, test_labels]]:
    pred = model.predict(test_images)
    pred_labels = tf.argmax(pred, axis=1)
    pred_labels = tf.cast(pred_labels, tf.int32)

    correct = tf.reduce_sum(tf.cast(pred_labels == labels, tf.int32))
    accuracy = correct / len(test_labels)

    print(f"Manual checking accuracy on {type} data: {accuracy * 100:.2f}%")
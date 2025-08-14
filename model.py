import os, random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers

seed=42
random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

try:
    from tensorflow.keras import mixed_precision
    if tf.config.list_physical_devices('GPU'):
        mixed_precision.set_global_policy('mixed_float16')
except Exception:
    pass

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

IMG_SIZE = 160
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1.0
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augment")

train = raw_train.map(format_example, num_parallel_calls=AUTOTUNE)
train = train.shuffle(SHUFFLE_BUFFER_SIZE)
train = train.map(lambda x,y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_batches = train.batch(BATCH_SIZE).prefetch(AUTOTUNE)

validation = raw_validation.map(format_example, num_parallel_calls=AUTOTUNE)
validation_batches = validation.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

test = raw_test.map(format_example, num_parallel_calls=AUTOTUNE)
test_batches = test.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, dtype='float32')

inputs = keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = keras.Model(inputs, outputs)

base_learning_rate = 1e-4
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        keras.metrics.AUC(from_logits=True, name="auc")
    ]
)

initial_epochs = 10
es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
ckpt_path = "best_mobilenetv2_cvdc.keras"
mc = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max")

history = model.fit(
    train_batches,
    epochs=initial_epochs,
    validation_data=validation_batches,
    callbacks=[es, mc]
)

base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate=base_learning_rate/10.0),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy', keras.metrics.AUC(from_logits=True, name="auc")]
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_batches,
    epochs=total_epochs,
    initial_epoch=len(history.history["loss"]),
    validation_data=validation_batches,
    callbacks=[es, mc]
)

best_model = keras.models.load_model(ckpt_path)
test_loss, test_acc, test_auc = best_model.evaluate(test_batches, verbose=0)
print({"test_loss": float(test_loss), "test_acc": float(test_acc), "test_auc": float(test_auc)})

acc = history.history.get('accuracy', []) + history_fine.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', []) + history_fine.history.get('val_accuracy', [])
loss = history.history.get('loss', []) + history_fine.history.get('loss', [])
val_loss = history.history.get('val_loss', []) + history_fine.history.get('val_loss', [])

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(acc, label='Eğitim Başarımı')
plt.plot(val_acc, label='Doğrulama Başarımı')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='İnce Ayar Başlangıcı')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Başarımı')

plt.subplot(2,1,2)
plt.plot(loss, label='Eğitim Kaybı')
plt.plot(val_loss, label='Doğrulama Kaybı')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='İnce Ayar Başlangıcı')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('epoch')
plt.show()

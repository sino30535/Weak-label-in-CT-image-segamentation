
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.applications import VGG16

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main(_):
    path = FLAGS.image_dir

    liver_path_list = []
    non_liver_path_list = []

    for file in os.listdir(os.path.join(path, 'liver')):
        if file.endswith(".png"):
            liver_path_list.append(os.path.join(path, 'liver', file))

    for file in os.listdir(os.path.join(path, 'noliver')):
        if file.endswith(".png"):
            non_liver_path_list.append(os.path.join(path, 'noliver', file))

    # Load images and create train, validation datasets
    label = np.array([0] * len(non_liver_path_list) + [1] * len(liver_path_list))
    data = np.zeros((len(non_liver_path_list) + len(liver_path_list), 224, 224, 3), dtype=np.uint8)

    with tqdm(range(len(non_liver_path_list)), ascii=True) as t:
        for i in t:
            img = cv2.imread(non_liver_path_list[i])
            img = img[:, :, ::-1]
            img = cv2.resize(img, (224, 224))
            data[i] = img
        t.close()

    with tqdm(range(len(liver_path_list)), ascii=True) as t:
        for i in t:
            img = cv2.imread(liver_path_list[i])
            img = img[:, :, ::-1]
            img = cv2.resize(img, (224, 224))
            data[i + len(non_liver_path_list)] = img
        t.close()

    x_train, x_val, y_train, y_val = train_test_split(data, label, shuffle=True, test_size=0.2, random_state=25)

    # Build GAP nad Dense layer on top of VGG16
    base_model = VGG16(include_top=False, weights='imagenet')
    for layers in base_model.layers:
        layers.trainable = False
    y = GlobalAveragePooling2D()(base_model.output)
    y = Dropout(0.25)(y)
    y = Dense(1, activation='sigmoid')(y)
    model = Model(inputs=base_model.input, outputs=y)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    # Specify model output path
    checkpoint_path = FLAGS.output_checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(x=x_train, y=y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              validation_data=(x_val, y_val),
              callbacks=[cp_callback],
              verbose=2)

    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./train256/',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_checkpoint',
        type=str,
        default='./output_model/checkpoint.ckpt',
        help='Where to save checkpoint.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size on training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Epochs on training'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.applications import VGG16
from tqdm import tqdm

import os
import sys
import cv2
import numpy as np
import argparse
import tensorflow as tf


def predict_on_weights(out_base, weights):
    gap = np.average(out_base, axis=(0, 1))
    logit = np.dot(gap, np.squeeze(weights))
    return 1 / (1 + np.e ** (-logit))


def pixel_segmentation(feature_maps, weights):
    predict = predict_on_weights(feature_maps, weights)
    # Weighted Feature Map
    cam = (predict - 0.5) * np.matmul(feature_maps, weights)
    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    # Resize as image size
    cam_resize = cv2.resize(cam, (256, 256))
    cam_resize = 255 * cam_resize
    cam_resize = cam_resize.astype(np.uint8)
    cam_resize[np.where(cam_resize <= FLAGS.sensitivity)] = 0
    cam_resize[np.where(cam_resize > FLAGS.sensitivity)] = 1
    output = np.zeros((256, 256, 3), dtype=np.uint8)
    if predict >= 0.5:
        for i in range(256):
            for j in range(256):
                if cam_resize[i][j] == 0:
                    output[i][j] = [255, 255, 255]
                elif cam_resize[i][j] == 1:
                    output[i][j] = [0, 128, 0]
    elif predict < 0.5:
        for i in range(256):
            for j in range(256):
                output[i][j] = [255, 255, 255]
    return output


def predict_single_image(base_model, image_path, weights):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    out_base = base_model.predict(np.expand_dims(img, axis=0))
    out_base = out_base[0]
    seg_output = pixel_segmentation(feature_maps=out_base, weights=weights)
    return seg_output


def main(_):
    base_model = VGG16(include_top=False, weights='imagenet')
    for layers in base_model.layers:
        layers.trainable = False
    y = GlobalAveragePooling2D()(base_model.output)
    y = Dropout(0.25)(y)
    y = Dense(1, activation='sigmoid')(y)
    model = Model(inputs=base_model.input, outputs=y)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_path = FLAGS.checkpoint_path

    model.load_weights(checkpoint_path)

    dense_layer_weights = model.layers[-1].get_weights()[0]

    image_dir = FLAGS.image_dir
    image_list = os.listdir(image_dir)
    image_path_list = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]

    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tqdm(range(len(image_path_list)), ascii=True) as t:
        for i in t:
            seg_output = predict_single_image(base_model, image_path_list[i], dense_layer_weights)
            cv2.imwrite(output_dir + "/" + image_list[i], seg_output)
        t.close()

    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./test256/images_pngs/',
        help='Path to folders of test set images.'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./output_model/checkpoint.ckpt',
        help='Checkpoint path.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output_prediction',
        help='Where to save prediction.'
    )
    parser.add_argument(
        '--sensitivity',
        type=int,
        default=130,
        help='Sensitivity level for prediction.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

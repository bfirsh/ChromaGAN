import os
import shutil
import sys
import tempfile
from pathlib import Path

import cog
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

sys.path.insert(0, "SOURCE")

import config as config
import dataClass as data
from ChromaGANPrint import deprocess, reconstruct_no


class Predictor(cog.Predictor):
    def setup(self):
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)

        config.BATCH_SIZE = 1
        save_path = os.path.join(config.MODEL_DIR, config.PRETRAINED)
        with self.graph.as_default(), self.sess.as_default():
            self.model = load_model(save_path)

    @cog.input("image", type=Path, help="Grayscale input image")
    def predict(self, image):
        input_dir = tempfile.mkdtemp()
        input_path = os.path.join(input_dir, image.name)
        shutil.copy(str(image), input_path)

        test_data = data.DATA(input_dir)
        batchX, _, _, original, labimg_oritList = test_data.generate_batch()
        with self.graph.as_default(), self.sess.as_default():
            predY, _ = self.model.predict(np.tile(batchX, [1, 1, 1, 3]))

        originalResult = original[0]
        height, width, _ = originalResult.shape
        predictedAB = cv2.resize(deprocess(predY[0]), (width, height))
        labimg_ori = np.expand_dims(labimg_oritList[0], axis=2)
        predResult = reconstruct_no(deprocess(labimg_ori), predictedAB)

        shutil.rmtree(input_dir)

        save_path = Path(tempfile.mkdtemp()) / "output.png"
        if not cv2.imwrite(str(save_path), predResult):
            raise Exception("Failed to save " + str(save_path))

        return save_path

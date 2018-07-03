import os
import boto3
import logging
import numpy as np
import model.utils as utils
import matplotlib.pyplot as plt


from tensorflow.contrib import keras
from model.decoder import final_model, s

L = keras.layers
K = keras.backend

IMG_SIZE = 299
BUCKET = 'lkimagecptioning'
PATH_WEIGHT = 'data/weights'
REGION_HOST = 's3.us-east-2.amazonaws.com'


class Model(object):
    def __init__(self):
        """
        Original weights are loaded from local folder, updated - from Amazon.
        """
        # self.params = self.load_weights_amazon('weights')
        self.nothing=0

    def load_weights_amazon(self, filename):
        """
        Load weights from Amazon.
        """
        s3 = boto3.client('s3',
                          aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                          aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        s3.download_file(BUCKET, filename, os.path.join(PATH_WEIGHT, filename))
        return os.path.join(PATH_WEIGHT, filename)

    # this is an actual prediction loop
    def generate_caption(self, image, t=1, sample=False, max_len=20):
        """
        Generate caption for given image.
        if `sample` is True, we will sample next token from predicted probability distribution.
        `t` is a temperature during that sampling,
            higher `t` causes more uniform-like distribution = more chaos.
        """

        # remember to reset your graph if you want to start building it from scratch!

        # final_models = final_model()
        # tf.reset_default_graph()
        # tf.set_random_seed(42)

        # condition lstm on the image
        s.run(final_model.init_lstm,
              {final_model.input_images: [image]})

        # current caption
        # start with only START token
        caption = [2]

        for _ in range(max_len):
            next_word_probs = s.run(final_model.one_step,
                                    {final_model.current_word: [caption[-1]]})[0]
            next_word_probs = next_word_probs.ravel()

            # apply temperature
            next_word_probs = next_word_probs**(1/t) / np.sum(next_word_probs**(1/t))

            if sample:
                next_word = np.random.choice(range(8769), p=next_word_probs)
            else:
                next_word = np.argmax(next_word_probs)

            caption.append(next_word)
            if next_word == 0:
                break
        vocab_inverse = utils.read_pickle('data/vocab_inverse.pkl')

        return list(map(vocab_inverse.get, caption))

    # look at validation prediction example
    def apply_model_to_image_raw_bytes(self, raw, filename):
        logging.info('Get')
        img = utils.decode_image_from_buf(raw)
        fig = plt.figure(figsize=(7, 7))
        plt.grid('off')
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('data/image/crop/{}'.format(filename), frameon=False,  bbox_inches='tight', pad_inches=0)
        img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)

        return ' '.join(self.generate_caption(image=img)[1:-1])

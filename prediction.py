# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from models import MobileNetV2

import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image

import argparse

parser = argparse.ArgumentParser('Prediction mnist label')

parser.add_argument('--path', type=str,
                    help='Image path, best input abs path. `./datasets/5.png`')
parser.add_argument('--height', type=int, default=224,
                    help='Image height. default: 224')
parser.add_argument('--width', type=int, default=224,
                    help='Image width.  default: 224')
parser.add_argument('--channels', type=int, default=3,
                    help='Image color RBG. default: 3')
parser.add_argument('--classes', type=int, default=2,
                    help="Classification picture type. default: 2")
parser.add_argument('--checkpoint_dir', '--dir', type=str, default='training_checkpoint',
                    help="Model save path.")
args = parser.parse_args()


label_name = {0: 'dog', 1: 'cat'}


def process_image(image, height=args.height, width=args.width):
  """ process image ops.

    Args:
      image: 'input tensor'.
      height: 'int64' img height.
      width: 'int64' img width.

    Returns:
      tensor

    """
  # read img to string.
  image = tf.io.read_file(image)
  # decode png to tensor
  image = tf.image.decode_image(image, channels=3)
  # convert image to float32
  image = tf.cast(image, tf.float32)
  # image norm.
  image = image / 255.
  # image resize model input size.
  image = tf.image.resize(image, (height, width))
  return image


def prediction(image):
  """ prediction image label.

  Args:
    image: 'input tensor'.

  Returns:
    'int64', label

  """
  image = process_image(image)
  # Add the image to a batch where it's the only member.
  image = (tf.expand_dims(image, 0))

  base_model = MobileNetV2(include_top=False,
                           input_shape=(args.height, args.width, args.channels),
                           weights=None,
                           classes=args.classes)

  avg_pool = tf.keras.layers.GlobalAveragePooling2D()
  fc = tf.keras.layers.Dense(args.classes,
                             activation=tf.nn.sigmoid,
                             use_bias=True,
                             name='Logits')

  model = tf.keras.Sequential([
    base_model,
    avg_pool,
    fc
  ])

  print(f"==========================================")
  print(f"Loading model.............................")
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir))
  print(f"Load model successful!")
  print(f"==========================================")
  print(f"Start making predictions about the picture.")
  print(f"==========================================")

  predictions = model(image)
  classes = tf.argmax(predictions[0])
  print(f"label is : {label_name[classes]}")

  image = Image.open(args.path)
  plt.figure(figsize=(4, 4))
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image, cmap='gray')
  plt.xlabel(int(classes))
  plt.show()


if __name__ == '__main__':
  prediction(args.path)

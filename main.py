import io
import os
import os.path
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
from PIL import Image

from model import DCGAN
from utils import pp, visualize, show_all_variables

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", None, "The size of batch images [64]")
flags.DEFINE_integer("grid_height", 8, "Grid Height")
flags.DEFINE_integer("grid_width", 8, "Grid Width")
flags.DEFINE_integer("input_height", None, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", None, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("sample_rate", None, "If == 5, it will take a sample image every 5 iterations")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_integer("nbr_of_layers_d", 5, "Number of layers in Discriminator")
flags.DEFINE_integer("nbr_of_layers_g", 5, "Number of layers in Generator")
flags.DEFINE_boolean("use_checkpoints", True, "Save and load checkpoints")
FLAGS = flags.FLAGS

# default batch_size
if FLAGS.batch_size is None and FLAGS.grid_height is not None and FLAGS.grid_width is not None:
  batch_size = FLAGS.grid_height * FLAGS.grid_width
elif FLAGS.batch_size is not None:
  batch_size = FLAGS.batch_size
else:
  raise Exception('grid_height/grid_width or batch_size must be provided')

# default size parameters
input_width = FLAGS.input_width
input_height = FLAGS.input_height
output_width = FLAGS.output_width
output_height = FLAGS.output_height

if (input_height is None and input_width is None) or (output_height is None and output_width is None):
  data_path = 'data/' + FLAGS.dataset
  first_image = [f for f in listdir(data_path) if isfile(join(data_path, f))][0]
  image_data = open(data_path + '/' + first_image, "rb").read()
  image = Image.open(io.BytesIO(image_data))
  rgb_im = image.convert('RGB')
  input_width = rgb_im.size[0]
  output_width = rgb_im.size[0]
  input_height = rgb_im.size[1]
  output_height = rgb_im.size[1]


def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if FLAGS.use_checkpoints and not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  sample_dir = FLAGS.sample_dir + "_g" + str(FLAGS.nbr_of_layers_g) + "_d" + str(FLAGS.nbr_of_layers_d)

  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth = True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
      sess,
      input_width=input_width,
      input_height=input_height,
      output_width=output_width,
      output_height=output_height,
      grid_height=FLAGS.grid_height,
      grid_width=FLAGS.grid_width,
      batch_size=batch_size,
      sample_num=batch_size,
      z_dim=FLAGS.generate_test_images,
      dataset_name=FLAGS.dataset,
      input_fname_pattern=FLAGS.input_fname_pattern,
      crop=FLAGS.crop,
      checkpoint_dir=FLAGS.checkpoint_dir,
      sample_dir=sample_dir,
      sample_rate=FLAGS.sample_rate,
      nbr_of_layers_d=FLAGS.nbr_of_layers_d,
      nbr_of_layers_g=FLAGS.nbr_of_layers_g,
      use_checkpoints=FLAGS.use_checkpoints)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    # Below is codes for visualization
    OPTION = 1
    visualize(sess, dcgan, FLAGS, batch_size, OPTION)


if __name__ == '__main__':
  tf.app.run()

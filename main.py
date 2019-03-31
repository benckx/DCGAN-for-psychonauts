import io
import os
import os.path

import numpy as np
import tensorflow as tf
from PIL import Image

from images_utils import get_images_recursively
from model import DCGAN
from utils import show_all_variables
from utils import visualize

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate_g", 0.0002, "Learning rate of for adam (G) [0.0002]")
flags.DEFINE_float("beta1_g", 0.5, "Momentum term of adam (G) [0.5]")
flags.DEFINE_float("learning_rate_d", 0.0002, "Learning rate of for adam (D) [0.0002]")
flags.DEFINE_float("beta1_d", 0.5, "Momentum term of adam (D) [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", None, "The size of batch images [64]")
flags.DEFINE_integer("grid_height", 8, "Grid Height")
flags.DEFINE_integer("grid_width", 8, "Grid Width")
flags.DEFINE_integer("input_height", None, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", None, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("sample_rate", None, "If == 5, it will take a sample image every 5 iterations")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [0]")
flags.DEFINE_integer("nbr_of_layers_g", 5, "Number of layers in Generator")
flags.DEFINE_integer("nbr_of_layers_d", 5, "Number of layers in Discriminator")
flags.DEFINE_boolean("use_checkpoints", False, "Save and load checkpoints")
flags.DEFINE_string("name", "dcgan", "Name of the job")
flags.DEFINE_boolean("batch_norm_g", False, "Batch normalization in Generator")
flags.DEFINE_boolean("batch_norm_d", False, "Batch normalization in Discriminator")
flags.DEFINE_string("activation_g", "relu", "Activation function in Generator")
flags.DEFINE_string("activation_d", "lrelu", "Activation function in Discriminator")
flags.DEFINE_integer("nbr_g_updates", 2, "Number of update of Generator optimizer (per iteration)")
flags.DEFINE_integer("nbr_d_updates", 1, "Number of update of Discriminator optimizer (per iteration)")
flags.DEFINE_integer("gpu_idx", None, "Index of GPU")
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
  first_image = get_images_recursively(data_path)[0]
  image_data = open(first_image, "rb").read()
  image = Image.open(io.BytesIO(image_data))
  rgb_im = image.convert('RGB')
  input_width = rgb_im.size[0]
  output_width = rgb_im.size[0]
  input_height = rgb_im.size[1]
  output_height = rgb_im.size[1]


def main(_):
  # pp.pprint(flags.FLAGS.__flags)
  print()
  print("FLAGS.nbr_of_layers_g: {}".format(FLAGS.nbr_of_layers_g))
  print("FLAGS.nbr_of_layers_d: {}".format(FLAGS.nbr_of_layers_d))
  print("FLAGS.dataset: {}".format(FLAGS.dataset))
  print("FLAGS.use_checkpoints: " + str(FLAGS.use_checkpoints))
  print("FLAGS.batch_norm_g: {}".format(FLAGS.batch_norm_g))
  print("FLAGS.batch_norm_d: {}".format(FLAGS.batch_norm_d))
  print("FLAGS.activation_g: {}".format(FLAGS.activation_g))
  print("FLAGS.activation_d: {}".format(FLAGS.activation_d))
  print("FLAGS.nbr_g_updates: {}".format(FLAGS.nbr_g_updates))
  print("FLAGS.nbr_d_updates: {}".format(FLAGS.nbr_d_updates))
  print()

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if FLAGS.use_checkpoints and not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  sample_dir = 'samples_' + FLAGS.name

  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  run_config.gpu_options.allow_growth = True
  run_config.gpu_options.per_process_gpu_memory_fraction = 0.9

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
      name=FLAGS.name,
      sample_rate=FLAGS.sample_rate,
      nbr_of_layers_d=FLAGS.nbr_of_layers_d,
      nbr_of_layers_g=FLAGS.nbr_of_layers_g,
      use_checkpoints=FLAGS.use_checkpoints,
      batch_norm_g=FLAGS.batch_norm_g,
      batch_norm_d=FLAGS.batch_norm_d,
      activation_g=FLAGS.activation_g.split(','),
      activation_d=FLAGS.activation_d.split(','),
      nbr_g_updates=FLAGS.nbr_g_updates,
      nbr_d_updates=FLAGS.nbr_d_updates,
      gpu_idx=FLAGS.gpu_idx)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    # Below is codes for visualization
    if FLAGS.generate_test_images > 0:
      OPTION = 1
      visualize(sess, dcgan, FLAGS, batch_size, OPTION, FLAGS.name)


if __name__ == '__main__':
  tf.app.run()

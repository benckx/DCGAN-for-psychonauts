from __future__ import division

import time
from glob import glob

from gpu_devices import GpuIterator
from ops import *
from utils import *


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
               batch_size=64, sample_num=64, output_height=64, output_width=64,
               grid_height=8, grid_width=8,
               y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
               gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
               input_fname_pattern='*.jpg', checkpoint_dir=None, name='dcgan', sample_dir=None, sample_rate=None,
               nbr_of_layers_d=5, nbr_of_layers_g=5, use_checkpoints=True, batch_norm_g=True, batch_norm_d=True,
               activation_g="relu", activation_d="lrelu", nbr_g_updates=2, nbr_d_updates=1):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.grid_width = grid_width
    self.grid_height = grid_height

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    self.name = name
    self.use_checkpoints = use_checkpoints
    self.sample_dir = sample_dir

    self.sample_rate = sample_rate
    self.nbr_of_layers_d = nbr_of_layers_d
    self.nbr_of_layers_g = nbr_of_layers_g

    self.batch_norm_g = batch_norm_g
    self.batch_norm_d = batch_norm_d

    self.activation_g = activation_g
    self.activation_d = activation_d

    self.nbr_g_updates = nbr_g_updates
    self.nbr_d_updates = nbr_d_updates

    self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
    np.random.shuffle(self.data)
    imread_img = imread(self.data[0])
    if len(imread_img.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(self.data[0]).shape[-1]
    else:
      self.c_dim = 1

    if len(self.data) < self.batch_size:
      raise Exception("[!] Entire dataset size is less than the configured batch_size")

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    gpu_iterator = GpuIterator()

    self.G = self.generator(self.z, gpu_iterator.next())
    self.D, self.D_logits = self.discriminator(inputs, gpu_iterator.next(), reuse=False)
    self.sampler = self.sampler(self.z, gpu_iterator.next())
    self.D_, self.D_logits_ = self.discriminator(self.G, gpu_iterator.next(), reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    print()
    print('self.nbr_g_updates: {}'.format(self.nbr_g_updates))
    print('self.nbr_d_updates: {}'.format(self.nbr_d_updates))
    print('self.activation_g: {}'.format(self.activation_g))
    print('self.activation_d: {}'.format(self.activation_d))
    print("config.learning_rate_g: {}".format(config.learning_rate_g))
    print("config.beta1_g: {}".format(config.beta1_g))
    print("config.learning_rate_d: {}".format(config.learning_rate_d))
    print("config.beta1_d: {}".format(config.beta1_d))
    print()

    g_optim = adam(config.learning_rate_g, config.beta1_g).minimize(self.g_loss, var_list=self.g_vars)
    d_optim = adam(config.learning_rate_d, config.beta1_d).minimize(self.d_loss, var_list=self.d_vars)

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

    sample_files = self.data[0:self.sample_num]
    sample = [
      get_image(sample_file,
                input_height=self.input_height,
                input_width=self.input_width,
                resize_height=self.output_height,
                resize_width=self.output_width,
                crop=self.crop,
                grayscale=self.grayscale) for sample_file in sample_files]
    if (self.grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)

    counter = 1
    start_time = time.time()
    if self.use_checkpoints:
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      self.data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
      np.random.shuffle(self.data)
      batch_idxs = min(len(self.data), config.train_size) // self.batch_size

      for idx in xrange(0, batch_idxs):
        batch_files = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [
          get_image(batch_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for batch_file in batch_files]
        if self.grayscale:
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        # Update D network
        for i in range(0, self.nbr_d_updates):
          _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.inputs: batch_images, self.z: batch_z})
          self.writer.add_summary(summary_str, counter)
          self.build_frame(i, epoch, idx, sample_z, sample_inputs)

        # Update G network
        # By default, run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        for i in range(0, self.nbr_g_updates):
          _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z})
          self.writer.add_summary(summary_str, counter)
          self.build_frame(self.nbr_d_updates + i, epoch, idx, sample_z, sample_inputs)

        errD_fake = self.d_loss_fake.eval({self.z: batch_z})
        errD_real = self.d_loss_real.eval({self.inputs: batch_images})
        errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

        # self.build_frame(self.nbr_d_updates + self.nbr_g_updates, epoch, idx, sample_z, sample_inputs)

        if self.use_checkpoints and np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def build_frame(self, suffix, epoch, idx, sample_z, sample_inputs):
    try:
      samples, d_loss, g_loss = self.sess.run(
        [self.sampler, self.d_loss, self.g_loss],
        feed_dict={
          self.z: sample_z,
          self.inputs: sample_inputs,
        },
      )
      file_name = './{}/train_{:06d}_{:06d}_{:03d}.png'.format(self.sample_dir, epoch, idx, suffix)
      save_images(samples, (self.grid_height, self.grid_width), file_name)
      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
    except Exception as e:
      print("one pic error! --> {}".format(e))

  def discriminator(self, image, device, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      nbr_layers = self.nbr_of_layers_d
      print('init discriminator with ' + str(nbr_layers) + ' layers ...')

      # layer 0
      previous_layer = conv2d(image, self.df_dim, name='d_h0_conv')
      previous_layer = add_activation(self.activation_d, previous_layer)

      # middle layers
      for i in range(1, nbr_layers - 1):
        output_dim = self.df_dim * (2 ** i)
        layer_name = 'd_h' + str(i) + '_conv'
        conv_layer = conv2d(previous_layer, output_dim, device, name=layer_name)
        if self.batch_norm_d:
          conv_layer = batch_norm(name='d_bn{}'.format(i))(conv_layer)
        previous_layer = add_activation(self.activation_d, conv_layer)

      # last layer
      layer_name = 'd_h' + str(nbr_layers - 1) + '_lin'
      last_layer = linear(tf.reshape(previous_layer, [self.batch_size, -1]), 1, layer_name)
      return tf.nn.sigmoid(last_layer), last_layer

  def generator(self, device, z):
    with tf.variable_scope("generator") as scope:
      nbr_layers = self.nbr_of_layers_g
      print('init generator with ' + str(nbr_layers) + ' layers ...')

      heights = []
      widths = []

      prev_h, prev_w = self.output_height, self.output_width
      heights.append(prev_h)
      widths.append(prev_w)

      for i in range(1, nbr_layers):
        prev_h, prev_w = conv_out_size_same(prev_h, 2), conv_out_size_same(prev_w, 2)
        heights.append(prev_h)
        widths.append(prev_w)

      mul = 2 ** (nbr_layers - 2)

      # layer 0
      height = heights[nbr_layers - 1]
      width = widths[nbr_layers - 1]
      z_ = linear(z, self.gf_dim * mul * height * width, 'g_h0_lin')
      prev_layer = tf.reshape(z_, [-1, heights[nbr_layers - 1], widths[nbr_layers - 1], self.gf_dim * mul])
      if self.batch_norm_g:
        prev_layer = batch_norm(name='g_bn0')(prev_layer)
      prev_layer = add_activation(self.activation_g, prev_layer)

      # middle layers
      for i in range(1, nbr_layers - 1):
        mul = mul // 2
        height = heights[nbr_layers - 1 - i]
        width = widths[nbr_layers - 1 - i]
        layer_name = 'g_h' + str(i)
        prev_layer = deconv2d(prev_layer, [self.batch_size, height, width, self.gf_dim * mul], device, name=layer_name)
        if self.batch_norm_g:
          prev_layer = batch_norm(name='g_bn' + str(i))(prev_layer)
        prev_layer = add_activation(self.activation_g, prev_layer)

      # last layer
      layer_name = 'g_h' + str(nbr_layers - 1)
      last_layer = deconv2d(prev_layer, [self.batch_size, heights[0], widths[0], self.c_dim], name=layer_name)

      return tf.nn.tanh(last_layer)

  def sampler(self, device, z):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      nbr_layers = self.nbr_of_layers_g

      heights = []
      widths = []

      prev_h, prev_w = self.output_height, self.output_width
      heights.append(prev_h)
      widths.append(prev_w)

      for i in range(1, nbr_layers):
        prev_h, prev_w = conv_out_size_same(prev_h, 2), conv_out_size_same(prev_w, 2)
        heights.append(prev_h)
        widths.append(prev_w)

      mul = 2 ** (nbr_layers - 2)

      # layer 0
      prev_layer = tf.reshape(
        linear(z, self.gf_dim * mul * heights[nbr_layers - 1] * widths[nbr_layers - 1], 'g_h0_lin'),
        [-1, heights[nbr_layers - 1], widths[nbr_layers - 1], self.gf_dim * mul])

      if self.batch_norm_g:
        prev_layer = batch_norm(name='g_bn0')(prev_layer, train=False)

      prev_layer = add_activation(self.activation_g, prev_layer)

      # middle layers
      for i in range(1, nbr_layers - 1):
        mul = mul // 2
        h = heights[nbr_layers - i - 1]
        w = widths[nbr_layers - i - 1]
        layer_name = 'g_h' + str(i)
        prev_layer = deconv2d(prev_layer, [self.batch_size, h, w, self.gf_dim * mul], device, name=layer_name)
        if self.batch_norm_g:
          prev_layer = batch_norm(name='g_bn' + str(i))(prev_layer, train=False)
        prev_layer = add_activation(self.activation_g, prev_layer)

      # last layer
      layer_name = 'g_h' + str(nbr_layers - 1)
      last_layer = deconv2d(prev_layer, [self.batch_size, heights[0], widths[0], self.c_dim], device, name=layer_name)
      return tf.nn.tanh(last_layer)

  @property
  def model_dir(self):
    return self.name

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0


def adam(learning_rate, beta1):
  """ Syntactic sugar """
  return tf.train.AdamOptimizer(learning_rate, beta1=beta1)


def add_activation(activation, layer):
  if activation == "relu":
    return tf.nn.relu(layer)
  elif activation == "lrelu":
    return tf.nn.leaky_relu(layer)
  elif activation == "elu":
    return tf.nn.elu(layer)
  elif activation == "tanh":
    return tf.nn.tanh(layer)
  elif activation == "sigmoid":
    return tf.nn.sigmoid(layer)
  else:
    print('Unknown activation ' + activation)
    exit(1)

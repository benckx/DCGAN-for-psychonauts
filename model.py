from __future__ import division

import datetime
import time
from multiprocessing import Pool

import images_utils
from dcgan_cmd_builder import Job
from files_utils import backup_checkpoint, must_backup_checkpoint, get_checkpoint_backup_delay
from gpu_devices import GpuAllocator
from images_utils import DataSetManager
from ops import *
from string_utils import min_to_string
from utils import *
from video_utils import get_box_name

frames_saving_pool = Pool(processes=20)
date_format = "%d/%m %H:%M"


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
  def __init__(self, sess, job: Job, input_height=108, input_width=108, crop=True,
               batch_size=64, sample_num=64, output_height=64, output_width=64,
               y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
               gfc_dim=1024, dfc_dim=1024, c_dim=3, input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, sample_rate=None,
               nbr_of_layers_d=5, nbr_of_layers_g=5, use_checkpoints=True, enable_cache=True):
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
    self.job = job

    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    self.use_checkpoints = use_checkpoints
    self.sample_dir = sample_dir

    self.sample_rate = sample_rate
    self.nbr_of_layers_d = nbr_of_layers_d
    self.nbr_of_layers_g = nbr_of_layers_g

    print('data folders: {}'.format(job.dataset_folders))
    self.data_set_manager = DataSetManager(job.dataset_folders, enable_cache, 'rgb')

    np.random.shuffle(self.data_set_manager.images_paths)
    imread_img = imread(self.data_set_manager.images_paths[0])
    if len(imread_img.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(self.data_set_manager.images_paths[0]).shape[-1]
    else:
      self.c_dim = 1

    if len(self.data_set_manager.images_paths) < self.batch_size:
      raise Exception("[!] Entire dataset size is less than the configured batch_size")

    self.grayscale = (self.c_dim == 1)
    self.frames_count = 0
    self.frames_last_timestamps = []

    self.gpu_allocator = GpuAllocator()

    print('generator device: {}'.format(self.gpu_allocator.generator_device()))
    print('sampler device: {}'.format(self.gpu_allocator.sampler_device()))
    print('discriminator device: {}'.format(self.gpu_allocator.discriminator_device()))
    print('discriminator fake device: {}'.format(self.gpu_allocator.discriminator_fake_device()))
    print('other device: {}'.format(self.gpu_allocator.other_things_device()))

    self.build_model()

  def build_model(self):
    if self.y_dim:
      with self.gpu_allocator.other_things_device():
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    with tf.device(self.gpu_allocator.discriminator_device()):
      self.inputs = tf.placeholder(
        tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    with tf.device(self.gpu_allocator.generator_device()):
      self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
      self.z_sum = histogram_summary("z", self.z)

    self.G = self.generator(self.z)
    self.D, self.D_logits = self.discriminator(inputs, reuse=False)
    self.sampler = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y, device):
      try:
        with tf.device(device):
          return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        with tf.device(device):
          return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    with tf.device(self.gpu_allocator.discriminator_device()):
      d_loss_real_input_tensor = sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D), self.gpu_allocator.discriminator_device())
      d_loss_fake_input_tensor = sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_), self.gpu_allocator.generator_device())

    with tf.device(self.gpu_allocator.generator_device()):
      g_loss_input_tensor = sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_), self.gpu_allocator.generator_device())

    with tf.device(self.gpu_allocator.discriminator_device()):
      self.d_loss_real = tf.reduce_mean(d_loss_real_input_tensor)
      self.d_loss_fake = tf.reduce_mean(d_loss_fake_input_tensor)

    with tf.device(self.gpu_allocator.generator_device()):
      self.g_loss = tf.reduce_mean(g_loss_input_tensor)

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    with tf.device(self.gpu_allocator.other_things_device()):
      self.saver = tf.train.Saver()

  def train(self, config):
    print()
    print('self.nbr_g_updates: {}'.format(self.job.nbr_g_updates))
    print('self.nbr_d_updates: {}'.format(self.job.nbr_d_updates))
    print('self.activation_g: {}'.format(self.job.activation_g))
    print('self.activation_d: {}'.format(self.job.activation_d))
    print("config.learning_rate_g: {}".format(config.learning_rate_g))
    print("config.beta1_g: {}".format(config.beta1_g))
    print("config.learning_rate_d: {}".format(config.learning_rate_d))
    print("config.beta1_d: {}".format(config.beta1_d))
    print("self.job.get_nbr_of_steps(): {}".format(self.job.get_nbr_of_steps()))
    print()

    with tf.device(self.gpu_allocator.generator_device()):
      g_optim = adam(config.learning_rate_g, config.beta1_g).minimize(self.g_loss, var_list=self.g_vars)

    with tf.device(self.gpu_allocator.discriminator_device()):
      d_optim = adam(config.learning_rate_d, config.beta1_d).minimize(self.d_loss, var_list=self.d_vars)

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

    counter = 1
    start_time = time.time()

    if self.use_checkpoints:
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")

    last_checkpoint_backup = int(time.time())
    checkpoint_backup_delay_in_min = get_checkpoint_backup_delay()

    self.job_start = datetime.datetime.now()

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
    sample_inputs = self.data_set_manager.get_random_images(self.sample_num)

    total_steps = self.job.get_nbr_of_steps()

    for step in xrange(total_steps):
      batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

      # Update D network
      for i in range(0, self.job.nbr_d_updates):
        batch_data = self.data_set_manager.get_random_images(self.batch_size)
        self.sess.run([d_optim, self.d_sum], feed_dict={self.inputs: batch_data, self.z: batch_z})

      self.build_frame(step, 0, sample_z, sample_inputs)

      # Update G network
      # By default, run g_optim twice to make sure that d_loss does not go to zero (different from paper)
      for i in range(0, self.job.nbr_g_updates):
        self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z})

      self.build_frame(step, 1, sample_z, sample_inputs)

      counter += 1
      print("Step: [%6d/%6d] time: %4.4f" % (step, total_steps, time.time() - start_time))

      if self.use_checkpoints and np.mod(counter, 500) == 2:
        try:
          begin = datetime.datetime.now().replace(microsecond=0)
          self.save(config.checkpoint_dir, counter)
          duration = datetime.datetime.now().replace(microsecond=0) - begin
          print('duration of checkpoint saving: {}'.format(duration))
          if must_backup_checkpoint():
            current_time = int(time.time())
            last_checkpoint_backup_min_ago = (current_time - last_checkpoint_backup) / 60
            print('last checkpoint backup: {:0.2f} min. ago'.format(last_checkpoint_backup_min_ago))
            if last_checkpoint_backup_min_ago >= checkpoint_backup_delay_in_min:
              print('time to save the thing')
              backup_checkpoint(self.job.name)
              last_checkpoint_backup = int(time.time())
            else:
              min_before_next_backup = checkpoint_backup_delay_in_min - last_checkpoint_backup_min_ago
              print('wait {:0.2f} more minutes before making a checkpoint backup'.format(min_before_next_backup))
        except Exception as e:
          print('Error during checkpoint saving: {}'.format(e))

  def build_frame(self, step, suffix, sample_z, sample_inputs):
    samples, d_loss, g_loss = self.sess.run(
      [self.sampler, self.d_loss, self.g_loss],
      feed_dict={
        self.z: sample_z,
        self.inputs: sample_inputs,
      },
    )

    if self.job.render_res is not None:
      box_grid_width = int(self.job.grid_width / (self.job.sample_res[0] / self.job.render_res[0]))
      box_grid_height = int(self.job.grid_height / (self.job.sample_res[1] / self.job.render_res[1]))
      box_grid_size = (box_grid_height, box_grid_width)

      nbr_of_boxes = images_utils.get_nbr_of_boxes(self.job.sample_res, self.job.render_res)
      tiles_per_box = int(len(samples) / nbr_of_boxes)

      for box_idx in range(1, nbr_of_boxes + 1):
        box_folder_name = '{}/{}'.format(self.sample_dir, get_box_name(box_idx))
        file_name = './{}/train_{:09d}_{:03d}.png'.format(box_folder_name, step, suffix)
        box_samples = samples[:tiles_per_box]
        save_images(box_samples, box_grid_size, file_name)
        # frames_saving_pool.apply_async(save_images, (box_samples, box_grid_size, file_name))
    else:
      file_name = './{}/train_{:09d}_{:03d}.png'.format(self.sample_dir, step, suffix)
      save_images(samples, (self.job.grid_height, self.job.grid_width), file_name)
      # frames_saving_pool.apply_async(save_images, (samples, (self.job.grid_height, self.job.grid_width), file_name))

    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
    self.log_performances(step)

  def log_performances(self, step):
    now = datetime.datetime.now()
    self.frames_count += 1
    self.frames_last_timestamps.append(now)
    min_since_started = ((now - self.job_start).seconds / 60)

    progress = step / self.job.get_nbr_of_steps()

    # calculate frame/min. on last 3 min.
    for value in self.frames_last_timestamps:
      delta = (now - value).seconds
      if delta >= 3 * 60:
        self.frames_last_timestamps.remove(value)

    print()
    if min_since_started >= 3:
      print('frames/min (3 min): {:0.2f}'.format(len(self.frames_last_timestamps) / 3))
    else:
      print('frames/min (total): {:0.2f}'.format(self.frames_count / min_since_started))

    # progress and time remaining estimate
    print("progress: {0:.2f}%".format(progress * 100))
    if progress >= 0.005:
      remaining_progress = 1 - progress
      remaining_time = (min_since_started / progress) * remaining_progress
      print('remaining time: ' + min_to_string(remaining_time))

      eta = now + datetime.timedelta(minutes=remaining_time)
      print('start: ' + self.job_start.strftime(date_format) + ' (' + min_to_string(min_since_started) + ' ago)')
      print('ETA: ' + eta.strftime(date_format))
      print()

  def discriminator(self, image, reuse=False):
    with tf.device(self.gpu_allocator.discriminator_device()):
      with tf.variable_scope("discriminator") as scope:
        if reuse:
          scope.reuse_variables()

        nbr_layers = self.nbr_of_layers_d
        print('init discriminator with ' + str(nbr_layers) + ' layers ...')

        # layer 0
        previous_layer = conv2d(image, self.df_dim, name='d_h0_conv', k_w=self.job.k_w, k_h=self.job.k_h)
        previous_layer = add_activation(self.job.activation_d[0], previous_layer)

        # middle layers
        for i in range(1, nbr_layers - 1):
          output_dim = self.df_dim * (2 ** i)
          layer_name = 'd_h' + str(i) + '_conv'
          conv_layer = conv2d(previous_layer, output_dim, name=layer_name, k_w=self.job.k_w, k_h=self.job.k_h)
          if self.job.batch_norm_d:
            conv_layer = batch_norm(name='d_bn{}'.format(i))(conv_layer)
          previous_layer = add_activation(self.job.activation_d[i], conv_layer)

        # last layer
        layer_name = 'd_h' + str(nbr_layers - 1) + '_lin'
        last_layer = linear(tf.reshape(previous_layer, [self.batch_size, -1]), 1, layer_name)
        return tf.nn.sigmoid(last_layer), last_layer

  def generator(self, z):
    with tf.device(self.gpu_allocator.generator_device()):
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
        if self.job.batch_norm_g:
          prev_layer = batch_norm(name='g_bn0')(prev_layer)
        prev_layer = add_activation(self.job.activation_g[0], prev_layer)

        # middle layers
        for i in range(1, nbr_layers - 1):
          mul = mul // 2
          height = heights[nbr_layers - 1 - i]
          width = widths[nbr_layers - 1 - i]
          layer_name = 'g_h' + str(i)
          output_shape = [self.batch_size, height, width, self.gf_dim * mul]
          prev_layer = deconv2d(prev_layer, output_shape, name=layer_name, k_w=self.job.k_w, k_h=self.job.k_h)
          if self.job.batch_norm_g:
            prev_layer = batch_norm(name='g_bn' + str(i))(prev_layer)
          prev_layer = add_activation(self.job.activation_g[i], prev_layer)

        # last layer
        layer_name = 'g_h' + str(nbr_layers - 1)
        output_shape = [self.batch_size, heights[0], widths[0], self.c_dim]
        last_layer = deconv2d(prev_layer, output_shape, name=layer_name, k_w=self.job.k_w, k_h=self.job.k_h)

        return tf.nn.tanh(last_layer)

  def sampler(self, z):
    with tf.device(self.gpu_allocator.sampler_device()):
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

        if self.job.batch_norm_g:
          prev_layer = batch_norm(name='g_bn0')(prev_layer, train=False)

        prev_layer = add_activation(self.job.activation_g[0], prev_layer)

        # middle layers
        for i in range(1, nbr_layers - 1):
          mul = mul // 2
          h = heights[nbr_layers - i - 1]
          w = widths[nbr_layers - i - 1]
          layer_name = 'g_h' + str(i)
          output_shape = [self.batch_size, h, w, self.gf_dim * mul]
          prev_layer = deconv2d(prev_layer, output_shape, name=layer_name, k_w=self.job.k_w, k_h=self.job.k_h)
          if self.job.batch_norm_g:
            prev_layer = batch_norm(name='g_bn' + str(i))(prev_layer, train=False)
          prev_layer = add_activation(self.job.activation_g[i], prev_layer)

        # last layer
        layer_name = 'g_h' + str(nbr_layers - 1)
        output_shape = [self.batch_size, heights[0], widths[0], self.c_dim]
        last_layer = deconv2d(prev_layer, output_shape, name=layer_name, k_w=self.job.k_w, k_h=self.job.k_h)
        return tf.nn.tanh(last_layer)

  @property
  def model_dir(self):
    return self.job.name

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
  if activation == "relu6":
    return tf.nn.relu6(layer)
  elif activation == "lrelu":
    return tf.nn.leaky_relu(layer)
  elif activation == "elu":
    return tf.nn.elu(layer)
  elif activation == "crelu":
    return tf.nn.crelu(layer)
  elif activation == "selu":
    return tf.nn.selu(layer)
  elif activation == "tanh":
    return tf.nn.tanh(layer)
  elif activation == "sigmoid":
    return tf.nn.sigmoid(layer)
  elif activation == "softplus":
    return tf.nn.softplus(layer)
  elif activation == "softsign":
    return tf.nn.softsign(layer)
  elif activation == "softmax":
    return tf.nn.softmax(layer)
  elif activation == "swish":
    return tf.nn.swish(layer)
  else:
    print('Unknown activation {}'.format(activation))
    exit(1)

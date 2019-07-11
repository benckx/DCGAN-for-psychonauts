import io
import os

import math
import numpy as np
import pandas as pd
from PIL import Image

from images_utils import get_datasets_images, get_boxes, get_nbr_of_boxes

samples_prefix = 'samples_'


def extend_array_to(input_array, nbr):
  result_array = input_array
  while len(input_array) < nbr:
    input_array.extend(input_array)
  return result_array[0:nbr]


class Job:

  def __init__(self):
    self.name = 'dcgan_job'
    self.batch_size = 0
    self.dataset_folders = []
    self.dataset_size = 0
    self.activation_g = ['relu']
    self.activation_d = ['lrelu']
    self.nbr_of_layers_g = 5
    self.nbr_of_layers_d = 5
    self.batch_norm_g = True
    self.batch_norm_d = True
    self.learning_rate_g = None
    self.learning_rate_d = None
    self.beta1_g = None
    self.beta1_d = None
    self.grid_width = 2
    self.grid_height = 2
    self.nbr_g_updates = 1
    self.nbr_d_updates = 1
    self.sample_folder = None
    self.render_video = True
    self.use_checkpoints = True
    self.delete_images_after_render = True
    self.upload_to_ftp = False
    self.has_auto_periodic_render = False
    self.sample_res = None
    self.render_res = None
    self.train_size = None
    self.video_length = None
    self.k_w = 5
    self.k_h = 5

  def get_nbr_of_frames(self):
    return self.get_nbr_of_steps() * 2

  def get_nbr_of_steps(self):
    return int(self.video_length * 1800)

  def compute_sample_res(self):
    dataset_images = get_datasets_images(self.dataset_folders)
    self.sample_folder = samples_prefix + self.name
    self.dataset_size = len(dataset_images)

    # assuming all input images have the same resolution
    first_image = dataset_images[0]
    image = Image.open(io.BytesIO(open(first_image, "rb").read()))
    rgb_im = image.convert('RGB')
    input_width = rgb_im.size[0]
    input_height = rgb_im.size[1]
    sample_width = self.grid_width * input_width
    sample_height = self.grid_height * input_height
    self.sample_res = (sample_width, sample_height)

  def get_boxes(self):
    if self.has_boxes():
      return get_boxes(self.sample_res, self.render_res)
    else:
      return []

  def get_nbr_of_boxes(self):
    if self.has_boxes():
      return get_nbr_of_boxes(self.sample_res, self.render_res)
    else:
      return 0

  def has_boxes(self):
    return not (self.render_res is None or self.sample_res == self.render_res)

  # noinspection PyListCreation
  def build_job_command(self, gpu_idx=None, enable_cache=True):
    dcgan_cmd = ['python3', 'main.py']

    dcgan_cmd.append('--video_length')
    dcgan_cmd.append(str(self.video_length))

    dcgan_cmd.append('--name')
    dcgan_cmd.append(self.name)

    dcgan_cmd.append('--dataset')
    dcgan_cmd.append(','.join(self.dataset_folders))

    dcgan_cmd.append('--grid_width')
    dcgan_cmd.append(str(self.grid_width))

    dcgan_cmd.append('--grid_height')
    dcgan_cmd.append(str(self.grid_height))

    dcgan_cmd.append('--nbr_of_layers_g')
    dcgan_cmd.append(str(self.nbr_of_layers_g))

    dcgan_cmd.append('--nbr_of_layers_d')
    dcgan_cmd.append(str(self.nbr_of_layers_d))

    if self.batch_norm_g:
      dcgan_cmd.append('--batch_norm_g')

    if self.batch_norm_d:
      dcgan_cmd.append('--batch_norm_d')

    if len(self.activation_g) > 0:
      dcgan_cmd.append('--activation_g')
      dcgan_cmd.append(','.join(self.activation_g))

    if len(self.activation_d) > 0:
      dcgan_cmd.append('--activation_d')
      dcgan_cmd.append(','.join(self.activation_d))

    if self.learning_rate_g is not None:
      dcgan_cmd.append('--learning_rate_g')
      dcgan_cmd.append(str(self.learning_rate_g))

    if self.beta1_g is not None:
      dcgan_cmd.append('--beta1_g')
      dcgan_cmd.append(str(self.beta1_g))

    if self.learning_rate_d is not None:
      dcgan_cmd.append('--learning_rate_d')
      dcgan_cmd.append(str(self.learning_rate_d))

    if self.beta1_d is not None:
      dcgan_cmd.append('--beta1_d')
      dcgan_cmd.append(str(self.beta1_d))

    dcgan_cmd.append('--nbr_g_updates')
    dcgan_cmd.append(str(self.nbr_g_updates))

    dcgan_cmd.append('--nbr_d_updates')
    dcgan_cmd.append(str(self.nbr_d_updates))

    dcgan_cmd.append('--k_w')
    dcgan_cmd.append(str(self.k_w))

    dcgan_cmd.append('--k_h')
    dcgan_cmd.append(str(self.k_h))

    if self.render_res is not None:
      dcgan_cmd.append('--render_res')
      dcgan_cmd.append('{}x{}'.format(self.render_res[0], self.render_res[1]))

    dcgan_cmd.append('--sample_rate')
    dcgan_cmd.append('1')

    if self.use_checkpoints:
      dcgan_cmd.append("--use_checkpoints")

    if gpu_idx is not None:
      dcgan_cmd.append("--gpu_idx")
      dcgan_cmd.append(str(gpu_idx))

    if not enable_cache:
      dcgan_cmd.append("--disable_cache")

    dcgan_cmd.append('--train')

    return dcgan_cmd

  def __str__(self):
    result = ''
    for key in self.__dict__.keys():
      result += '{} -> {}\n'.format(key, self.__dict__[key])

    result += 'get_nbr_of_steps() -> {}'.format(self.get_nbr_of_steps())

    return result


  @classmethod
  def from_row(cls, row, columns):
    # model settings
    job = Job()
    job.name = row['name']
    job.grid_width = int(row['grid_width'])
    job.grid_height = int(row['grid_height'])
    job.batch_size = job.grid_width * job.grid_height

    if 'learning_rate_g' in columns and str(row['learning_rate_g']) != 'nan':
      job.learning_rate_g = float(row['learning_rate_g'])

    if 'beta1_g' in columns and str(row['beta1_g']) != 'nan':
      job.beta1_g = float(row['beta1_g'])

    if 'learning_rate_d' in columns and str(row['learning_rate_d']) != 'nan':
      job.learning_rate_d = float(row['learning_rate_d'])

    if 'beta1_d' in columns and str(row['beta1_d']) != 'nan':
      job.beta1_d = float(row['beta1_d'])

    # layers
    job.nbr_of_layers_d = int(row['nbr_of_layers_d'])
    job.nbr_of_layers_g = int(row['nbr_of_layers_g'])

    if 'batch_norm_g' in columns:
      job.batch_norm_g = row['batch_norm_g']

    if 'batch_norm_d' in columns:
      job.batch_norm_d = row['batch_norm_d']

    if 'activation_g' in columns and str(row['activation_g']) != 'nan':
      job.activation_g = row['activation_g'].split(',')

    if 'activation_d' in columns and str(row['activation_d']) != 'nan':
      job.activation_d = row['activation_d'].split(',')

    job.activation_d = extend_array_to(job.activation_d, job.nbr_of_layers_d - 1)
    job.activation_g = extend_array_to(job.activation_g, job.nbr_of_layers_g - 1)

    if 'k_w' in columns:
      job.k_w = int(row['k_w'])

    if 'k_h' in columns:
      job.k_h = int(row['k_h'])

    # input images
    job.dataset_folders = row['dataset'].split(',')
    job.compute_sample_res()

    # video settings
    if 'render_video' in columns:
      job.render_video = row['render_video']

    if 'nbr_g_updates' in columns:
      job.nbr_g_updates = int(row['nbr_g_updates'])

    if 'nbr_d_updates' in columns:
      job.nbr_d_updates = int(row['nbr_d_updates'])

    if row['video_length'] and not math.isnan(row['video_length']):
      job.video_length = float(row['video_length'])

    # periodic renders
    if job.render_video:
      try:
        job.has_auto_periodic_render = row['auto_render_period']
        if job.has_auto_periodic_render:
          job.auto_render_period = int(row['auto_render_period'])
      except:
        job.has_auto_periodic_render = False

      if job.has_auto_periodic_render:
        if 'render_res' in columns and str(row['render_res']) != '' and str(row['render_res']) != 'nan':
          job.render_res = tuple([int(x) for x in row['render_res'].split('x')])

    # flags
    if 'upload_to_ftp' in columns:
      job.upload_to_ftp = row['upload_to_ftp']

    if 'delete_images_after_render' in columns:
      job.delete_images_after_render = row['delete_images_after_render']

    if 'use_checkpoints' in columns:
      job.use_checkpoints = row['use_checkpoints']

    return job

  @classmethod
  def from_csv_file(cls, csv_file):
    data = pd.read_csv(csv_file, encoding='UTF-8')

    # parse jobs
    jobs = []
    for _, row in data.iterrows():
      print(str(row))
      jobs.append(Job.from_row(row, data.columns))

    return jobs

  @classmethod
  def validate(cls, jobs):
    # validate ftp
    for job in jobs:
      if job.upload_to_ftp and not os.path.exists('ftp.ini'):
        print('option upload_to_ftp == true but ftp.ini file was not found')
        exit(1)

    # validate names uniqueness
    names = []
    for job in jobs:
      names.append(job.name)

    if len(names) != len(set(names)):
      print('Names are not unique')
      exit(1)

    # validate datasets
    data_folders = [f for f in os.listdir('data/')]

    config_file_datasets = []
    for job in jobs:
      config_file_datasets.extend(job.dataset_folders)

    for dataset in config_file_datasets:
      if dataset not in data_folders:
        print('Error: dataset {} not found!'.format(dataset))
        exit(1)

  # noinspection PyPep8Naming
  @classmethod
  def from_FLAGS(cls, FLAGS):
    job = Job()
    job.name = FLAGS.name
    job.nbr_of_layers_d = FLAGS.nbr_of_layers_d
    job.nbr_of_layers_g = FLAGS.nbr_of_layers_g
    job.use_checkpoints = FLAGS.use_checkpoints
    job.batch_norm_g = FLAGS.batch_norm_g
    job.batch_norm_d = FLAGS.batch_norm_d
    job.activation_g = FLAGS.activation_g.split(',')
    job.activation_d = FLAGS.activation_d.split(',')
    job.nbr_g_updates = FLAGS.nbr_g_updates
    job.nbr_d_updates = FLAGS.nbr_d_updates
    job.grid_height = FLAGS.grid_height
    job.grid_width = FLAGS.grid_width
    job.learning_rate_g = FLAGS.learning_rate_g
    job.learning_rate_d = FLAGS.learning_rate_d
    job.beta1_g = FLAGS.beta1_g
    job.beta1_d = FLAGS.beta1_d
    job.train_size = np.inf
    job.dataset_folders = FLAGS.dataset.split(',')
    job.video_length = FLAGS.video_length
    job.k_w = FLAGS.k_w
    job.k_h = FLAGS.k_h
    if FLAGS.render_res is not None:
      job.render_res = tuple([int(val) for val in FLAGS.render_res.split('x')])
    job.compute_sample_res()

    return job

  @classmethod
  def must_start_auto_periodic_renders(cls, jobs):
    auto_periodic_renders = False

    for job in jobs:
      if not auto_periodic_renders:
        auto_periodic_renders = job.has_auto_periodic_render

    return auto_periodic_renders

import io
import os

import math
import pandas as pd
from PIL import Image

from images_utils import get_datasets_images


class Job:

  def __init__(self):
    self.name = 'dcgan_job'
    self.epochs = 0
    self.batch_size = 0
    self.dataset_folders = []
    self.dataset_size = 0
    self.activation_g = []
    self.activation_d = []
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
    self.use_checkpoints = False
    self.delete_images_after_render = False
    self.upload_to_ftp = False
    self.has_auto_periodic_render = False
    self.sample_res = None
    self.render_res = None

  def get_nbr_of_frames(self):
    frames_per_step = 2
    return frames_per_step * int(self.dataset_size / self.batch_size) * self.epochs

  # noinspection PyListCreation
  def build_job_command(self, gpu_idx=None, enable_cache=True):
    dcgan_cmd = ['python3', 'main.py']

    dcgan_cmd.append('--epoch')
    dcgan_cmd.append(str(self.epochs))

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

  @classmethod
  def from_row(cls, row):
    fps = 60
    samples_prefix = 'samples_'

    # model settings
    job = Job()
    job.name = row['name']
    job.epochs = int(row['epoch'])
    job.grid_width = int(row['grid_width'])
    job.grid_height = int(row['grid_height'])
    job.batch_size = job.grid_width * job.grid_height

    if row['learning_rate_g'] and not math.isnan(row['learning_rate_g']):
      job.learning_rate_g = float(row['learning_rate_g'])

    if row['beta1_g']:
      job.beta1_g = float(row['beta1_g'])

    if row['learning_rate_d'] and not math.isnan(row['learning_rate_d']):
      job.learning_rate_d = float(row['learning_rate_d'])

    if row['beta1_d']:
      job.beta1_d = float(row['beta1_d'])

    # layers
    job.nbr_of_layers_d = int(row['nbr_of_layers_d'])
    job.nbr_of_layers_g = int(row['nbr_of_layers_g'])

    job.batch_norm_g = row['batch_norm_g'] == '' or row['batch_norm_g']
    job.batch_norm_d = row['batch_norm_d'] == '' or row['batch_norm_d']

    if row['activation_g'] and str(row['activation_g']) != "nan":
      job.activation_g = row['activation_g'].split(',')

    if row['activation_d'] and str(row['activation_d']) != "nan":
      job.activation_d = row['activation_d'].split(',')

    # input images
    job.dataset_folders = row['dataset'].split(',')
    job.dataset_images = get_datasets_images(job.dataset_folders)
    job.sample_folder = samples_prefix + job.name
    job.dataset_size = len(job.dataset_images)

    # assuming all input images have the same resolution
    first_image = job.dataset_images[0]
    image = Image.open(io.BytesIO(open(first_image, "rb").read()))
    rgb_im = image.convert('RGB')
    input_width = rgb_im.size[0]
    input_height = rgb_im.size[1]
    job.sample_width = job.grid_width * input_width
    job.sample_height = job.grid_height * input_height
    job.sample_res = (job.sample_width, job.sample_height)

    # video settings
    job.render_video = row['render_video']

    if row['nbr_g_updates'] and not math.isnan(row['nbr_g_updates']):
      job.nbr_g_updates = int(row['nbr_g_updates'])
    else:
      job.nbr_g_updates = 2

    if row['nbr_d_updates'] and not math.isnan(row['nbr_d_updates']):
      job.nbr_d_updates = int(row['nbr_d_updates'])
    else:
      job.nbr_d_updates = 1

    job.video_length_in_min = ((job.get_nbr_of_frames() / fps) / 60)

    # periodic renders
    if job.render_video:
      job.has_auto_periodic_render = row['auto_render_period'] and row['auto_render_period'] > 0
      if job.has_auto_periodic_render:
        job.auto_render_period = int(row['auto_render_period'])

      if job.has_auto_periodic_render:
        if row['render_res'] and str(row['render_res']) != '' and str(row['render_res']) != 'nan':
          job.render_res = tuple([int(x) for x in row['render_res'].split('x')])

    # flags
    job.upload_to_ftp = row['upload_to_ftp']
    job.delete_images_after_render = row['delete_images_after_render']
    job.use_checkpoints = row['use_checkpoints']

    return job

  @classmethod
  def from_csv_file(cls, csv_file):
    data = pd.read_csv(csv_file, encoding='UTF-8')

    # parse jobs
    jobs = []
    for _, row in data.iterrows():
      print(str(row))
      jobs.append(Job.from_row(row))

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

  @classmethod
  def must_start_auto_periodic_renders(cls, jobs):
    auto_periodic_renders = False

    for job in jobs:
      if not auto_periodic_renders:
        auto_periodic_renders = job.has_auto_periodic_render

    return auto_periodic_renders

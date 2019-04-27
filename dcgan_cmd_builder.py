import io

import math
from PIL import Image

from images_utils import get_datasets_images


# noinspection PyListCreation
def build_dcgan_cmd(cmd_row, gpu_idx, enable_cache):
  dcgan_cmd = ["python3", "main.py"]

  dcgan_cmd.append("--epoch")
  dcgan_cmd.append(str(cmd_row['epoch']))

  dcgan_cmd.append("--name")
  dcgan_cmd.append(cmd_row['name'])

  dcgan_cmd.append("--dataset")
  dcgan_cmd.append(cmd_row['dataset'])

  dcgan_cmd.append("--grid_width")
  dcgan_cmd.append(str(cmd_row['grid_width']))
  dcgan_cmd.append("--grid_height")
  dcgan_cmd.append(str(cmd_row['grid_height']))

  if cmd_row['nbr_of_layers_g']:
    dcgan_cmd.append("--nbr_of_layers_g")
    dcgan_cmd.append(str(cmd_row['nbr_of_layers_g']))

  if cmd_row['nbr_of_layers_d']:
    dcgan_cmd.append("--nbr_of_layers_d")
    dcgan_cmd.append(str(cmd_row['nbr_of_layers_d']))

  if cmd_row['batch_norm_g'] == '' or cmd_row['batch_norm_g']:
    dcgan_cmd.append("--batch_norm_g")

  if cmd_row['batch_norm_g'] == '' or cmd_row['batch_norm_d']:
    dcgan_cmd.append("--batch_norm_d")

  if cmd_row['activation_g'] and str(cmd_row['activation_g']) != "nan":
    dcgan_cmd.append("--activation_g")
    dcgan_cmd.append(cmd_row['activation_g'])

  if cmd_row['activation_d'] and str(cmd_row['activation_d']) != "nan":
    dcgan_cmd.append("--activation_d")
    dcgan_cmd.append(cmd_row['activation_d'])

  if cmd_row['learning_rate_g'] and not math.isnan(cmd_row['learning_rate_g']):
    dcgan_cmd.append("--learning_rate_g")
    dcgan_cmd.append(str(cmd_row['learning_rate_g']))

  if cmd_row['beta1_g'] and not math.isnan(cmd_row['beta1_g']):
    dcgan_cmd.append("--beta1_g")
    dcgan_cmd.append(str(cmd_row['beta1_g']))

  if cmd_row['learning_rate_d'] and not math.isnan(cmd_row['learning_rate_d']):
    dcgan_cmd.append("--learning_rate_d")
    dcgan_cmd.append(str(cmd_row['learning_rate_d']))

  if cmd_row['beta1_d'] and not math.isnan(cmd_row['beta1_d']):
    dcgan_cmd.append("--beta1_d")
    dcgan_cmd.append(str(cmd_row['beta1_d']))

  if cmd_row['nbr_g_updates'] and not math.isnan(cmd_row['nbr_g_updates']):
    dcgan_cmd.append('--nbr_g_updates')
    dcgan_cmd.append(str(int(cmd_row['nbr_g_updates'])))

  if cmd_row['nbr_d_updates'] and not math.isnan(cmd_row['nbr_d_updates']):
    dcgan_cmd.append('--nbr_d_updates')
    dcgan_cmd.append(str(int(cmd_row['nbr_d_updates'])))

  if cmd_row['use_checkpoints']:
    dcgan_cmd.append("--use_checkpoints")

  dcgan_cmd.append('--sample_rate')
  dcgan_cmd.append('1')

  if gpu_idx is not None:
    dcgan_cmd.append("--gpu_idx")
    dcgan_cmd.append(str(gpu_idx))

  if not enable_cache:
    dcgan_cmd.append("--disable_cache")

  dcgan_cmd.append('--train')

  return dcgan_cmd


class Job:

  def __init__(self):
    self.epochs = 0
    self.batch_size = 0
    self.dataset_size = 0

  def get_nbr_of_frames(self):
    frames_per_step = 2
    return frames_per_step * int(self.dataset_size / self.batch_size) * self.epochs

  @classmethod
  def from_row(cls, row):
    job = Job()
    job.name = row['name']
    job.grid_width = int(row['grid_width'])
    job.grid_height = int(row['grid_height'])

    job.dataset_folders = row['dataset'].split(',')
    job.dataset_images = get_datasets_images(job.dataset_folders)

    job.epochs = int(row['epoch'])

    if row['nbr_g_updates'] and not math.isnan(row['nbr_g_updates']):
      job.nbr_g_updates = int(row['nbr_g_updates'])
    else:
      job.nbr_g_updates = 2

    if row['nbr_d_updates'] and not math.isnan(row['nbr_d_updates']):
      job.nbr_d_updates = int(row['nbr_d_updates'])
    else:
      job.nbr_d_updates = 1

    first_image = job.dataset_images[0]
    image = Image.open(io.BytesIO(open(first_image, "rb").read()))
    rgb_im = image.convert('RGB')
    input_width = rgb_im.size[0]
    input_height = rgb_im.size[1]
    job.sample_width = job.grid_width * input_width
    job.sample_height = job.grid_height * input_height

    job.batch_size = job.grid_width * job.grid_height

    job.has_auto_periodic_renders = row['auto_render_period'] and row['auto_render_period'] > 0
    if job.has_auto_periodic_renders:
      job.auto_render_period = int(row['auto_render_period'])

    if job.has_auto_periodic_renders:
      if row['render_res'] and str(row['render_res']) != '' and str(row['render_res']) != 'nan':
        job.render_res = tuple([int(x) for x in row['render_res'].split('x')])
      else:
        job.render_res = None
    else:
      job.render_res = None

    job.dataset_size = len(job.dataset_images)

    fps = 60
    job.sample_res = (job.sample_width, job.sample_height)
    job.video_length_in_min = ((job.get_nbr_of_frames() / fps) / 60)

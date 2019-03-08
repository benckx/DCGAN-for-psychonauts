import configparser
import datetime
import ftplib
import io
import math
import os.path
import shutil
import subprocess
import threading
import traceback
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join

import pandas as pd
from PIL import Image


# noinspection PyListCreation
def build_dcgan_cmd(cmd_row):
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

  if cmd_row['learning_rate'] and not math.isnan(cmd_row['learning_rate']):
    dcgan_cmd.append("--learning_rate")
    dcgan_cmd.append(str(cmd_row['learning_rate']))

  if cmd_row['beta1'] and not math.isnan(cmd_row['beta1']):
    dcgan_cmd.append("--beta1")
    dcgan_cmd.append(str(cmd_row['beta1']))

  if cmd_row['nbr_g_updates'] and not math.isnan(cmd_row['nbr_g_updates']):
    dcgan_cmd.append("--nbr_g_updates")
    dcgan_cmd.append(str(int(cmd_row['nbr_g_updates'])))

  if cmd_row['use_checkpoints']:
    dcgan_cmd.append("--use_checkpoints")

  dcgan_cmd.append("--sample_rate")
  dcgan_cmd.append("1")

  dcgan_cmd.append("--train")

  return dcgan_cmd


# noinspection PyListCreation
def render_video(name, images_folder):
  ffmpeg_cmd = ['ffmpeg']
  ffmpeg_cmd.append('-framerate')
  ffmpeg_cmd.append('30')
  ffmpeg_cmd.append('-f')
  ffmpeg_cmd.append('image2')
  ffmpeg_cmd.append('-i')
  ffmpeg_cmd.append(images_folder + '/%*.png')
  ffmpeg_cmd.append('-c:v')
  ffmpeg_cmd.append('libx264')
  ffmpeg_cmd.append('-profile:v')
  ffmpeg_cmd.append('high')
  ffmpeg_cmd.append('-crf')
  ffmpeg_cmd.append('17')
  ffmpeg_cmd.append('-pix_fmt')
  ffmpeg_cmd.append('yuv420p')
  ffmpeg_cmd.append(name + '.mp4')
  subprocess.run(ffmpeg_cmd)


def process_video(video_name, images_folder, upload_to_ftp, delete_images):
  """ Render to video, upload to ftp """

  render_video(video_name, images_folder)

  if upload_to_ftp:
    try:
      config = configparser.ConfigParser()
      config.read('ftp.ini')
      session = ftplib.FTP(config['ftp']['host'], config['ftp']['user'], config['ftp']['password'])
      file = open(video_name + '.mp4', 'rb')
      session.storbinary('STOR ' + video_name + '.mp4', file)
      file.close()
      session.quit()
    except Exception as exception:
      print('error during FTP transfer -> {}'.format(exception))

  if delete_images:
    shutil.rmtree(images_folder)


def scheduled_job():
  try:
    global current_cut
    print('------ periodic render ------')

    if os.path.exists(sample_folder):
      sample_folder_size = len([f for f in listdir(sample_folder) if isfile(join(data_path, f))])
      print('{} sample folder size: {}'.format(sample_folder, sample_folder_size))
      print('current cut: {}'.format(current_cut))
      print('auto_periodic_render_frames: {}'.format(auto_periodic_renders_nbr_of_frames))
      print('we do: {}'.format(auto_periodic_renders_nbr_of_frames > sample_folder_size))
      if auto_periodic_renders_nbr_of_frames > sample_folder_size:
        create_video_cut()
        current_cut += 1
  except NameError:
    print("sample_folder not defined yet")
  finally:
    print('----- / periodic render -----')
    threading.Timer(60.0, scheduled_job).start()


def create_video_cut():
  frames = [f for f in listdir(sample_folder) if isfile(join(data_path, f))][0:auto_periodic_renders_nbr_of_frames + 1]
  cut_folder = '{}_cut{:02d}'.format(sample_folder, current_cut)
  video_name = job_name + '_cut{:02d}'.format(current_cut)
  os.makedirs(cut_folder)
  for f in frames:
    print('moving ' + f)
    os.rename(sample_folder + '/' + f, cut_folder + '/' + f)
  process_video(video_name, current_cut, True, True)


fps = 30
samples_prefix = 'samples_'
data_folders = [f for f in listdir('data/')]
csv_files = [f for f in listdir('.') if (isfile(join('.', f)) and f.endswith(".csv"))]
csv_files.sort()

if len(csv_files) == 0:
  print('Error: no csv file')
  exit(1)

print('found config file: ' + csv_files[0])
print()

data = pd.read_csv(csv_files[0], encoding='UTF-8')

# validate ftp
for index, row in data.iterrows():
  if row['upload_to_ftp'] and not os.path.exists('ftp.ini'):
    print('option upload_to_ftp == true but ftp.ini file was not found')
    exit(1)

# validate names
names = []
for index, row in data.iterrows():
  names.append(row['name'])

if (len(names)) != len(set(names)):
  print('Names are not unique')
  exit(1)

# validate datasets
for index, row in data.iterrows():
  if row['dataset'] not in data_folders:
    print('Error: dataset ' + row['dataset'] + ' not found!')
    exit(1)

pool = Pool(processes=10)

# determine if we do automatic periodic renders
auto_periodic_renders = False

for index, row in data.iterrows():
  if not auto_periodic_renders:
    auto_periodic_renders = row['auto_render_period'] and row['auto_render_period'] > 0

if auto_periodic_renders:
  pool.apply_async(scheduled_job)

# run the jobs
for index, row in data.iterrows():
  print(str(row))
  try:
    data_path = 'data/' + row['dataset']
    first_image = [f for f in listdir(data_path) if isfile(join(data_path, f))][0]
    image = Image.open(io.BytesIO(open(data_path + '/' + first_image, "rb").read()))
    rgb_im = image.convert('RGB')
    input_width = rgb_im.size[0]
    input_height = rgb_im.size[1]
    sample_width = row['grid_width'] * input_width
    sample_height = row['grid_height'] * input_height
    dataset_size = len(listdir(data_path))
    batch_size = row['grid_width'] * row['grid_height']
    nbr_of_frames = int(dataset_size / batch_size) * row['epoch']
    sample_folder = samples_prefix + row['name']

    if auto_periodic_renders:
      auto_periodic_renders_nbr_of_frames = row['auto_render_period'] * fps
      job_name = row['name']
      current_cut = 1

    print('')
    print('sample resolution: {}x{}'.format(sample_width, sample_height))
    print('dataset size: {}'.format(dataset_size))
    print('total nbr. of frames: {}'.format(nbr_of_frames))
    print('length of video: {} min.'.format((nbr_of_frames / fps) / 60))
    print('frames per minutes: {}'.format(fps * 60))
    print('automatic periodic render: {}'.format(auto_periodic_renders))
    if auto_periodic_renders:
      auto_periodic_renders_nbr_of_frames = row['auto_render_period'] * fps
      print('nbr of frames in periodic renders: {}'.format(auto_periodic_renders_nbr_of_frames))
    print('')

    begin = datetime.datetime.now().replace(microsecond=0)
    job_cmd = build_dcgan_cmd(row)
    print('command: ' + ' '.join('{}'.format(v) for v in job_cmd))
    subprocess.run(job_cmd)
    duration = datetime.datetime.now().replace(microsecond=0) - begin
    print('duration of the job: {}'.format(duration))

    # process video asynchronously
    if not auto_periodic_renders and row['render_video']:
      pool.apply_async(process_video, (row['name'], row['upload_to_ftp'], row['delete_images_after_render']))
  except Exception as e:
    print('error during process of {} -> {}'.format(row['name'], e))
    print(traceback.format_exc())

pool.close()
pool.join()

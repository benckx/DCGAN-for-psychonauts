import configparser
import ftplib
import os.path
import subprocess
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join

import pandas as pd

samples_prefix = 'samples_'
data_folders = [f for f in listdir('data/')]
csv_files = [f for f in listdir('.') if (isfile(join('.', f)) and f.endswith(".csv"))]


def video_rendered():
  print('video process asynchronously')


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


def process_video(name, upload_to_ftp, delete_images):
  """ Render to video, upload to ftp """

  sample_folder = samples_prefix + name
  render_video(name, sample_folder)

  if upload_to_ftp:
    try:
      config = configparser.ConfigParser()
      config.read('ftp.ini')
      session = ftplib.FTP(config['vjloops']['host'], config['vjloops']['user'], config['vjloops']['password'])
      file = open(name + '.mp4', 'rb')
      session.storbinary('STOR ' + name + '.mp4', file)
      file.close()
      session.quit()
    except ValueError as e:
      print('error during FTP transfer: ' + str(e))

  if delete_images:
    os.rmdir(sample_folder)


if len(csv_files) == 0:
  print('Error: no csv file')
  exit(1)

data = pd.read_csv(csv_files[0], encoding='UTF-8')

# validate datasets
for index, row in data.iterrows():
  if row['dataset'] not in data_folders:
    print('Error: dataset ' + row['dataset'] + ' not found !')
    exit(1)

pool = Pool(processes=10)

for index, row in data.iterrows():
  print(str(row))

  # noinspection PyListCreation
  dcgan_cmd = ["python3", "main.py"]

  dcgan_cmd.append("--epoch")
  dcgan_cmd.append(str(row['epoch']))

  dcgan_cmd.append("--name")
  dcgan_cmd.append(str(row['name']))

  dcgan_cmd.append("--dataset")
  dcgan_cmd.append(str(row['dataset']))

  dcgan_cmd.append("--grid_width")
  dcgan_cmd.append(str(row['grid_width']))
  dcgan_cmd.append("--grid_height")
  dcgan_cmd.append(str(row['grid_height']))

  dcgan_cmd.append("--nbr_of_layers_g")
  dcgan_cmd.append(str(row['nbr_of_layers_g']))
  dcgan_cmd.append("--nbr_of_layers_d")
  dcgan_cmd.append(str(row['nbr_of_layers_d']))

  if row['batch_norm_d']:
    dcgan_cmd.append("--batch_norm_d")

  if row['batch_norm_g']:
    dcgan_cmd.append("--batch_norm_g")

  dcgan_cmd.append("--sample_rate")
  dcgan_cmd.append("1")

  dcgan_cmd.append("--train")

  subprocess.run(dcgan_cmd)

  # process video asynchronously
  if row['render_video']:
    pool.apply_async(process_video, (row['name'], row['delete_images_after_render'], row['upload_to_ftp']))
else:
  print('Config file not found')

pool.close()
pool.join()

import configparser
import ftplib
import shutil
import subprocess
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join

import pandas as pd

samples_prefix = 'samples_'
data_folders = [f for f in listdir('data/')]
csv_files = [f for f in listdir('.') if (isfile(join('.', f)) and f.endswith(".csv"))]
csv_files.sort()


# noinspection PyListCreation
def build_dcgan_cmd(cmd_row):
  dcgan_cmd = ["python3", "main.py"]

  dcgan_cmd.append("--epoch")
  dcgan_cmd.append(str(cmd_row['epoch']))

  dcgan_cmd.append("--name")
  dcgan_cmd.append(str(cmd_row['name']))

  dcgan_cmd.append("--dataset")
  dcgan_cmd.append(str(cmd_row['dataset']))

  dcgan_cmd.append("--grid_width")
  dcgan_cmd.append(str(cmd_row['grid_width']))
  dcgan_cmd.append("--grid_height")
  dcgan_cmd.append(str(cmd_row['grid_height']))

  dcgan_cmd.append("--nbr_of_layers_g")
  dcgan_cmd.append(str(cmd_row['nbr_of_layers_g']))
  dcgan_cmd.append("--nbr_of_layers_d")
  dcgan_cmd.append(str(cmd_row['nbr_of_layers_d']))

  if cmd_row['batch_norm_d']:
    dcgan_cmd.append("--batch_norm_d")

  if cmd_row['batch_norm_g']:
    dcgan_cmd.append("--batch_norm_g")

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


def process_video(name, upload_to_ftp, delete_images):
  """ Render to video, upload to ftp """

  sample_folder = samples_prefix + name
  render_video(name, sample_folder)

  if upload_to_ftp:
    try:
      config = configparser.ConfigParser()
      config.read('ftp.ini')
      session = ftplib.FTP(config['ftp']['host'], config['ftp']['user'], config['ftp']['password'])
      file = open(name + '.mp4', 'rb')
      session.storbinary('STOR ' + name + '.mp4', file)
      file.close()
      session.quit()
    except:
      print('error during FTP transfer')

  if delete_images:
    shutil.rmtree(sample_folder)


if len(csv_files) == 0:
  print('Error: no csv file')
  exit(1)

print('found config file: ' + csv_files[0])
print()

data = pd.read_csv(csv_files[0], encoding='UTF-8')

# validate names
names = []
for index, row in data.iterrows():
  names.append(row['name'])

if (len(names)) != len(set(names)):
  print('Names are not unique')

# validate datasets
for index, row in data.iterrows():
  if row['dataset'] not in data_folders:
    print('Error: dataset ' + row['dataset'] + ' not found!')
    exit(1)

pool = Pool(processes=10)

for index, row in data.iterrows():
  print(str(row))
  try:
    job_cmd = build_dcgan_cmd(row)
    subprocess.run(job_cmd)

    # process video asynchronously
    print('render video: ' + str(row['render_video']))
    if row['render_video']:
      pool.apply_async(process_video, (row['name'], row['upload_to_ftp'], row['delete_images_after_render']))
  except:
    print('error during process of: ' + row['name'])

pool.close()
pool.join()

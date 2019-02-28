import subprocess
from os import listdir
from os.path import isfile, join

import pandas as pd


def render_video(name):
  # noinspection PyListCreation
  ffmpeg_cmd = ['ffmpeg']
  ffmpeg_cmd.append('-framerate')
  ffmpeg_cmd.append('30')
  ffmpeg_cmd.append('-f')
  ffmpeg_cmd.append('image2')
  ffmpeg_cmd.append('-i')
  ffmpeg_cmd.append('samples_' + name + '/%*.png')
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


data_folders = [f for f in listdir('data/')]
csv_files = [f for f in listdir('.') if (isfile(join('.', f)) and f.endswith(".csv"))]

if len(csv_files) != 1:
  print('csv issues: ' + str(csv_files))
  exit(1)

data = pd.read_csv(csv_files[0], encoding='UTF-8')

for index, row in data.iterrows():
  if row['dataset'] not in data_folders:
    print('Error: dataset ' + row['dataset'] + ' not found !')
    exit(1)

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

  if row['process_video']:
    render_video(row['name'])
else:
  print('Config file not found')


# import threading
# thread = threading.Thread(target=f)
# thread.start()

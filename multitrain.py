import datetime
import io
import os.path
import subprocess
import sys
import traceback
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from os import listdir
from os.path import isfile, join

import pandas as pd
from PIL import Image

import images_utils
from dcgan_cmd_builder import *
from files_utils import backup_checkpoint, must_backup_checkpoint
from images_utils import get_datasets_images
from shared_state import ThreadsSharedState
from video_utils import process_video, periodic_render_job


class MyManager(BaseManager):
  pass


BaseManager.register('ThreadsSharedState', ThreadsSharedState)
manager = BaseManager()
manager.start()

# define constants
fps = 60
samples_prefix = 'samples_'
data_folders = [f for f in listdir('data/')]
csv_files = [f for f in listdir('.') if (isfile(join('.', f)) and f.endswith(".csv"))]
csv_files.sort()

pool = Pool(processes=10)

csv_file = None
gpu_idx = None
enable_cache = True

if len(sys.argv) > 1:
  params = sys.argv[0:]
  for idx, param in enumerate(params):
    if param == '--config':
      csv_file = params[idx + 1]
      print('csv file == {}'.format(params[idx + 1]))
    if param == '--gpu_idx':
      gpu_idx = params[idx + 1]
      print('gpu_idx == {}'.format(params[idx + 1]))
    if param == '--disable_cache':
      enable_cache = False

# validate csv config file
if csv_file is None:
  if len(csv_files) == 0:
    print('Error: no csv file')
    exit(1)

  csv_file = csv_files[0]
  print('found config file: ' + csv_file)
else:
  print('config file passed in param: ' + csv_file)

print()

data = pd.read_csv(csv_file, encoding='UTF-8')

# validate ftp
for _, row in data.iterrows():
  if row['upload_to_ftp'] and not os.path.exists('ftp.ini'):
    print('option upload_to_ftp == true but ftp.ini file was not found')
    exit(1)

# validate names
names = []
for _, row in data.iterrows():
  names.append(row['name'])

if len(names) != len(set(names)):
  print('Names are not unique')
  exit(1)

# validate datasets
config_file_datasets = []
for _, row in data.iterrows():
  config_file_datasets.extend(row['dataset'].split(','))

for dataset in config_file_datasets:
  if dataset not in data_folders:
    print('Error: dataset {} not found!'.format(dataset))
    exit(1)

# determine if we do automatic periodic renders
auto_periodic_renders = False

for index, row in data.iterrows():
  if not auto_periodic_renders:
    auto_periodic_renders = row['auto_render_period'] and row['auto_render_period'] > 0

# launch schedule job if needed
shared_state = None
if auto_periodic_renders:
  # noinspection PyUnresolvedReferences
  shared_state = manager.ThreadsSharedState()
  pool.apply(periodic_render_job, args=[shared_state, True])

# run the jobs
for _, row in data.iterrows():
  print(str(row))
  try:
    dataset_images = get_datasets_images(row['dataset'].split(','))
    first_image = dataset_images[0]
    image = Image.open(io.BytesIO(open(first_image, "rb").read()))
    rgb_im = image.convert('RGB')
    input_width = rgb_im.size[0]
    input_height = rgb_im.size[1]
    sample_width = row['grid_width'] * input_width
    sample_height = row['grid_height'] * input_height

    dataset_size = len(dataset_images)
    batch_size = row['grid_width'] * row['grid_height']
    frames_per_iteration = 0

    if row['nbr_g_updates'] and not math.isnan(row['nbr_g_updates']):
      frames_per_iteration += int(row['nbr_g_updates'])
    else:
      frames_per_iteration += 2

    if row['nbr_d_updates'] and not math.isnan(row['nbr_d_updates']):
      frames_per_iteration += int(row['nbr_d_updates'])
    else:
      frames_per_iteration += 1

    nbr_of_frames = frames_per_iteration * int(dataset_size / batch_size) * row['epoch']
    sample_res = (sample_width, sample_height)
    render_res = None
    video_length_in_min = ((nbr_of_frames / fps) / 60)

    print('')
    if auto_periodic_renders and row['render_video']:
      shared_state.init_current_cut()
      shared_state.set_folder(samples_prefix + row['name'])
      shared_state.set_job_name(row['name'])
      shared_state.set_frames_threshold(row['auto_render_period'] * fps)
      shared_state.set_upload_to_ftp(row['upload_to_ftp'])
      shared_state.set_delete_at_the_end(row['delete_images_after_render'])
      print('frames threshold: {}'.format(shared_state.get_frames_threshold()))
      print('sample folder: {}'.format(shared_state.get_folder()))

    print('dataset size: {}'.format(dataset_size))
    print('video length: {} frames --> {:0.2f} min.'.format(nbr_of_frames, video_length_in_min))
    print('frames per minutes: {}'.format(fps * 60))
    print('automatic periodic render: {}'.format(auto_periodic_renders))
    print('sample resolution: {}'.format(sample_res))
    if row['render_res'] and str(row['render_res']) != '' and str(row['render_res']) != 'nan':
      render_res = tuple([int(x) for x in row['render_res'].split('x')])
      print('render resolution: {}'.format(render_res))
      print('boxes: {}'.format(images_utils.get_boxes(sample_res, render_res)))
      if auto_periodic_renders:
        shared_state.set_sample_res(sample_res)
        shared_state.set_render_res(render_res)
    print('')

    begin = datetime.datetime.now().replace(microsecond=0)
    job_cmd = build_dcgan_cmd(row, gpu_idx, enable_cache)
    print('command: ' + ' '.join('{}'.format(v) for v in job_cmd))
    process = subprocess.run(job_cmd)
    print('return code: {}'.format(process.returncode))
    duration = datetime.datetime.now().replace(microsecond=0) - begin
    print('duration of the job {} -> {}'.format(row['name'], duration))

    if process.returncode == 0:
      sample_folder = samples_prefix + row['name']
      upload_to_ftp = row['upload_to_ftp']
      delete_after = row['delete_images_after_render']

      # process video async
      if row['render_video']:
        if auto_periodic_renders:
          os.rename(sample_folder, shared_state.get_time_cut_folder_name())
          sample_folder = shared_state.get_time_cut_folder_name()

        pool.apply_async(process_video, (sample_folder, upload_to_ftp, delete_after, sample_res, render_res))

      # backup checkpoint one last time
      if must_backup_checkpoint() and row['use_checkpoint']:
        backup_checkpoint(row['name'])
  except Exception as e:
    print('error during process of {} -> {}'.format(row['name'], e))
    print(traceback.format_exc())

pool.close()
pool.join()

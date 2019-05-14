import datetime
import subprocess
import sys
import traceback
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from os import listdir
from os.path import isfile, join

import images_utils
from dcgan_cmd_builder import *
from files_utils import backup_checkpoint, must_backup_checkpoint
from shared_state import ThreadsSharedState
from video_utils import process_video_job_param, periodic_render_job


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

# parse and validate jobs
jobs = Job.from_csv_file(csv_file)
Job.validate(jobs)

# launch schedule job if needed
auto_periodic_renders = Job.must_start_auto_periodic_renders(jobs)
shared_state = None
if auto_periodic_renders:
  # noinspection PyUnresolvedReferences
  shared_state = manager.ThreadsSharedState()
  pool.apply(periodic_render_job, args=[shared_state, True])

# run the jobs
for job in jobs:
  try:
    print('')
    if job.has_auto_periodic_render:
      shared_state.init_current_cut()
      shared_state.set_folder(job.sample_folder)
      shared_state.set_job_name(job.name)
      shared_state.set_frames_threshold(job.auto_render_period * fps)
      shared_state.set_upload_to_ftp(job.upload_to_ftp)
      shared_state.set_delete_at_the_end(job.delete_images_after_render)
      print('frames threshold: {}'.format(shared_state.get_frames_threshold()))
      print('sample folder: {}'.format(shared_state.get_folder()))

    print('dataset size: {}'.format(job.dataset_size))
    print('video length: {:0.2f} min.'.format(job.video_length))
    print('frames per minutes: {}'.format(fps * 60))
    print('automatic periodic render: {}'.format(auto_periodic_renders))
    print('sample resolution: {}'.format(job.sample_res))
    if job.render_res is not None:
      print('render resolution: {}'.format(job.render_res))
      print('boxes: {}'.format(images_utils.get_boxes(job.sample_res, job.render_res)))
      if job.has_auto_periodic_render:
        shared_state.set_sample_res(job.sample_res)
        shared_state.set_render_res(job.render_res)
    print('')

    begin = datetime.datetime.now().replace(microsecond=0)
    job_cmd = job.build_job_command(gpu_idx, enable_cache)
    print('command: ' + ' '.join('{}'.format(v) for v in job_cmd))
    process = subprocess.run(job_cmd)
    print('return code: {}'.format(process.returncode))
    duration = datetime.datetime.now().replace(microsecond=0) - begin
    print('duration of the job {} -> {}'.format(job.name, duration))

    # process video async
    if process.returncode == 0:
      if job.render_video:
        if auto_periodic_renders:
          os.rename(job.sample_folder, shared_state.get_time_cut_folder_name())
          job.sample_folder = shared_state.get_time_cut_folder_name()

        # pool.apply_async(process_video_job_param, job)
        process_video_job_param(job)

      # backup checkpoint one last time
      if must_backup_checkpoint() and job.use_checkpoints:
        backup_checkpoint(job.name)
  except Exception as e:
    print('error during process of {} -> {}'.format(job.name, e))
    print(traceback.format_exc())

pool.close()
pool.join()

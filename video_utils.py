import os.path
import os.path
import shutil
import subprocess
import threading
from os import listdir
from os.path import isfile, join

from PIL import Image

import images_utils
from files_utils import upload_via_ftp
from shared_state import ThreadsSharedState


# noinspection PyListCreation
def render_video(images_folder):
  ffmpeg_cmd = ['ffmpeg']
  ffmpeg_cmd.append('-framerate')
  ffmpeg_cmd.append('60')
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
  ffmpeg_cmd.append(images_folder + '.mp4')
  subprocess.run(ffmpeg_cmd)


def process_video(images_folder, upload_to_ftp, delete_images, sample_res=None, render_res=None):
  """ Crop images, render to video, upload to ftp """

  if sample_res is None or render_res is None or sample_res == render_res:
    render_video(images_folder)
    if upload_to_ftp:
      print('Sending {}.mp4 to ftp'.format(images_folder))
      upload_via_ftp(images_folder + '.mp4')
  else:
    box_idx = 1
    for box in images_utils.get_boxes(sample_res, render_res):
      box_folder_name = '{}_box{:04d}'.format(images_folder, box_idx)
      print('Box folder: {}'.format(box_folder_name))
      os.makedirs(box_folder_name)

      original_frames = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
      original_frames.sort()
      for f in original_frames:
        src = images_folder + '/' + f
        dest = box_folder_name + '/' + f
        print('Extracting {} to {} with {}'.format(src, dest, box))
        region = Image.open(src).crop(box)
        region.save(dest)

      render_video(box_folder_name)

      if upload_to_ftp:
        print('Sending {}.mp4 to ftp'.format(box_folder_name))
        upload_via_ftp(box_folder_name + '.mp4')

      if delete_images:
        shutil.rmtree(box_folder_name)

      box_idx += 1

  if delete_images:
    shutil.rmtree(images_folder)


def periodic_render_job(shared: ThreadsSharedState, loop=True):
  print()
  print('------ periodic render ------')
  if shared is not None:
    print('sample folder: {}'.format(shared.get_folder()))
    print('current time cut: {}'.format(shared.get_current_cut()))
    print('frame threshold: {}'.format(shared.get_frames_threshold()))
    print('sample resolution: {}'.format(shared.get_sample_res()))
    print('render resolution: {}'.format(shared.get_render_res()))
    print('loop at the end: {}'.format(loop))
    print('')

    if shared.get_folder() is not None and os.path.exists(shared.get_folder()):
      folder_size = len([f for f in listdir(shared.get_folder()) if isfile(join(shared.get_folder(), f))])
      print('{} folder size: {}'.format(shared.get_folder(), folder_size))
      print('proceed to time cut: {}'.format(shared.get_frames_threshold() < folder_size))
      if shared.get_frames_threshold() < folder_size:
        create_video_time_cut(shared)
        shared.increment_cut()
    else:
      print('folder does not exist yet')
  else:
    print('shared state not defined yet')

  print('----- / periodic render -----')
  print()

  if loop:
    threading.Timer(120.0, periodic_render_job, args=[shared]).start()


def create_video_time_cut(shared: ThreadsSharedState):
  nbr_frames = shared.get_frames_threshold()
  folder = shared.get_folder()
  frames = [f for f in listdir(folder) if isfile(join(folder, f))]
  frames.sort()
  time_cut_folder = '{}_time_cut{:04d}'.format(shared.get_job_name(), shared.get_current_cut())
  print('Time cut folder: {}'.format(time_cut_folder))
  os.makedirs(time_cut_folder)
  for f in frames[0:nbr_frames]:
    src = shared.get_folder() + '/' + f
    dest = time_cut_folder + '/' + f
    print('moving from {} to {}'.format(src, dest))
    os.rename(src, dest)

  upload_to_ftp = shared.is_upload_to_ftp()
  delete_at_the_end = shared.is_delete_at_the_end()
  sample_res = shared.get_sample_res()
  render_res = shared.get_render_res()
  process_video(time_cut_folder, upload_to_ftp, delete_at_the_end, sample_res, render_res)

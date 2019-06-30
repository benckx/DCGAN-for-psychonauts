import os.path
import os.path
import shutil
import subprocess
import threading
from os import listdir
from os.path import isfile, join

import images_utils
from dcgan_cmd_builder import Job
from files_utils import upload_via_ftp
from shared_state import ThreadsSharedState


def get_box_name(box_idx):
  return 'box{:04d}'.format(box_idx)


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


def process_video_job_param(job: Job):
  print('job.sample_folder: ' + job.sample_folder)
  print('job.sample_res: {}'.format(job.sample_res))
  print('job.render_res: {}'.format(job.render_res))
  process_video(job.sample_folder, job.upload_to_ftp, job.delete_images_after_render, job.sample_res, job.render_res)


def process_video(images_folder, upload_to_ftp, delete_images, sample_res=None, render_res=None):
  """ Render to video, upload to ftp """

  if sample_res is None or render_res is None or sample_res == render_res:
    render_video(images_folder)
    video_file_name = images_folder + '.mp4'

    if upload_to_ftp:
      print('Sending {} to ftp'.format(video_file_name))
      upload_via_ftp(video_file_name)

    os.rename(video_file_name, '../renders/' + video_file_name)
  else:
    original_frames = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
    original_frames.sort()

    boxes = images_utils.get_boxes(sample_res, render_res)
    for box_idx in range(1, len(boxes) + 1):
      box_folder_name = get_box_name(box_idx)
      render_video(box_folder_name)
      video_file_name = box_folder_name + '.mp4'

      if upload_to_ftp:
        print('Sending {} to ftp'.format(video_file_name))
        upload_via_ftp(video_file_name)

      os.rename(video_file_name, '../renders/' + video_file_name)

  if delete_images:
    shutil.rmtree(images_folder)


def periodic_render_job(shared: ThreadsSharedState, loop=True):
  print()
  print('------ periodic render ------')
  if shared is not None:
    print('current time cut: {}'.format(shared.get_current_cut()))
    print('frame threshold: {}'.format(shared.get_frames_threshold()))
    print('loop at the end: {}'.format(loop))
    print('')

    proceed = must_proceed_time_cut(shared)
    print('proceed to time cut: {}'.format(proceed))

    if proceed:
      create_video_time_cut(shared)
      shared.increment_cut()
  else:
    print('shared state not defined yet')

  print('----- / periodic render -----')
  print()

  if loop:
    threading.Timer(30.0, periodic_render_job, args=[shared]).start()


def must_proceed_time_cut(shared: ThreadsSharedState):
  if shared.get_folder() is not None and os.path.exists(shared.get_folder()):
    if not shared.has_boxes():
      # TODO: move to function
      folder_size = len([f for f in listdir(shared.get_folder()) if isfile(join(shared.get_folder(), f))])
      print('{} folder size: {}'.format(shared.get_folder(), folder_size))
      return shared.get_frames_threshold() < folder_size
    else:
      boxes = images_utils.get_boxes(shared.get_sample_res(), shared.get_render_res())
      for box_idx in range(1, len(boxes) + 1):
        box_folder_name = shared.get_folder() + '/' + get_box_name(box_idx)
        # TODO: move to function
        box_folder_size = len([f for f in listdir(box_folder_name) if isfile(join(box_folder_name, f))])
        print('{} size: {}'.format(box_folder_name, box_folder_size))
        if shared.get_frames_threshold() > box_folder_size:
          return False

      return True
  else:
    return False


def create_video_time_cut(shared: ThreadsSharedState):
  nbr_frames = shared.get_frames_threshold()
  folder = shared.get_folder()
  frames = [f for f in listdir(folder) if isfile(join(folder, f))]
  frames.sort()
  time_cut_folder = shared.get_time_cut_folder_name()
  print('Time cut folder: {}'.format(time_cut_folder))
  os.makedirs(time_cut_folder)

  upload_to_ftp = shared.is_upload_to_ftp()
  delete_at_the_end = shared.is_delete_at_the_end()

  if shared.has_boxes():
    boxes = images_utils.get_nbr_of_boxes(shared.get_sample_res(), shared.get_render_res())
    for box_idx in range(1, len(boxes) + 1):
      box_folder_name = get_box_name(box_idx)
      target_box_folder = '{}/{}'.format(time_cut_folder, box_folder_name)
      os.makedirs(target_box_folder)
      for f in frames[0:nbr_frames]:
        src = shared.get_folder() + '/' + box_folder_name + '/' + f
        dest = target_box_folder + '/' + f
        print('moving from {} to {}'.format(src, dest))
        os.rename(src, dest)

      process_video(target_box_folder, upload_to_ftp, delete_at_the_end, shared.sample_res, shared.render_res)
  else:
    for f in frames[0:nbr_frames]:
      # TODO: refactor
      src = shared.get_folder() + '/' + f
      dest = time_cut_folder + '/' + f
      print('moving from {} to {}'.format(src, dest))
      os.rename(src, dest)

    process_video(time_cut_folder, upload_to_ftp, delete_at_the_end)

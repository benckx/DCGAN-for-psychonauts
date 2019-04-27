import os
import subprocess
import sys

import math


def get_video_duration_seconds(file_name):
  command = "ffmpeg -i " + file_name + " 2>&1 | grep Duration | cut -d ' ' -f 4 | sed s/,//"
  result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
  duration = result.stdout.decode("utf-8")
  split = duration.replace('.', ':').split(':')
  minutes = int(split[1])
  seconds = int(split[2])
  return seconds + (60 * minutes)


def seconds_to_timestamp(seconds: int):
  minutes = math.floor(seconds / 60)
  seconds = seconds - (60 * minutes)

  minutes_padded = str(minutes).rjust(2, '0')
  seconds_padded = str(seconds).rjust(2, '0')

  return '00:' + minutes_padded + ':' + seconds_padded


def main():
  file_path = sys.argv[1]
  length = 60
  re_encode = False

  if len(sys.argv) > 1:
    if len(sys.argv) > 1:
      params = sys.argv[0:]
      for idx, param in enumerate(params):
        if param == '--file':
          file_path = params[idx + 1]
        if param == '--length':
          length = int(params[idx + 1])
        if param == '--reencode':
          re_encode = True

  file_name = file_path.split('/')[-1]
  file_folder = file_path[0:file_path.find(file_name)]
  file_name_no_extension = file_name.split('.')[0]

  print(file_path)
  print(file_name)
  print(file_name_no_extension)
  print(file_folder)

  endpos = seconds_to_timestamp(length)

  count = 1
  for i in range(0, get_video_duration_seconds(file_path) - length, length):
    ss = seconds_to_timestamp(i)
    cut_num = str(count).rjust(3, '0')
    output_file = file_folder + file_name_no_extension + '_' + cut_num + '.mp4'
    command = 'mencoder -ss ' + ss + ' -endpos ' + endpos + ' -oac copy -ovc copy ' + file_path + ' -o ' + output_file
    subprocess.run(command, shell=True)
    if re_encode:
      re_encode_output = file_folder + file_name_no_extension + '_' + cut_num + '_encoded' + '.mp4'
      encoding = '-c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p'
      re_encode_command = 'ffmpeg -i ' + output_file + ' ' + encoding + ' ' + re_encode_output
      subprocess.run(re_encode_command, shell=True)
      os.remove(output_file)
    count += 1


if __name__ == "__main__":
  main()

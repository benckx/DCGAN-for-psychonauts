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
  offset = 0
  re_encode = False

  if len(sys.argv) > 1:
    if len(sys.argv) > 1:
      params = sys.argv[0:]
      for idx, param in enumerate(params):
        if param == '--file':
          file_path = params[idx + 1]
        elif param == '--length':
          length = int(params[idx + 1])
        elif param == '--offset':
          offset = int(params[idx + 1])
        elif param == '--reencode':
          re_encode = True
        elif param.startswith('--'):
          print('param {} not known'.format(param))
          exit(0)

  file_name = file_path.split('/')[-1]
  file_folder = file_path[0:file_path.find(file_name)]
  file_name_no_extension = file_name.split('.')[0]
  duration = get_video_duration_seconds(file_path)
  endpos = seconds_to_timestamp(length)
  nbr_of_bits = math.floor((duration - offset) / length)

  print('length: {}'.format(length))
  print('offset: {}'.format(offset))
  print(file_path)
  print(file_name)
  print(file_name_no_extension)
  print(file_folder)
  print(duration)
  print(nbr_of_bits)

  count = 1
  for i in range(nbr_of_bits):
    start = (i * length) + offset
    end = ((i + 1) * length) + offset
    print("bit {} --> from {} to {}".format(i, start, end))
    count += 1

  count = 1
  for i in range(nbr_of_bits):
    ss = seconds_to_timestamp((i * length) + offset)
    cut_num = str(count).rjust(3, '0')
    count += 1
    output_file = file_folder + file_name_no_extension + '_' + cut_num + '.mp4'
    command = 'mencoder -ss ' + ss + ' -endpos ' + endpos + ' -oac copy -ovc copy ' + file_path + ' -o ' + output_file
    subprocess.run(command, shell=True)
    if re_encode:
      re_encode_output = file_folder + file_name_no_extension + '_' + cut_num + '_encoded' + '.mp4'
      encoding = '-c:v libx264 -profile:v high -crf 17 -pix_fmt yuv420p'
      re_encode_command = 'ffmpeg -i ' + output_file + ' ' + encoding + ' ' + re_encode_output
      subprocess.run(re_encode_command, shell=True)
      os.remove(output_file)


if __name__ == "__main__":
  main()

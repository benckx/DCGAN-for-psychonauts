import subprocess
import sys
from os import listdir
from os.path import isfile, join

folder = sys.argv[1]

time_cuts_files = [f for f in listdir(folder) if (isfile(join(folder, f)) and f.find('_time_cut') > -1)]
time_cuts_files.sort()

boxes_files = [f for f in listdir(folder) if (isfile(join(folder, f)) and f.find('_box') > -1)]
boxes_files.sort()

boxes_names = set()
for box_file in boxes_files:
  box_with_extension = box_file[box_file.find('_box') + 1:]
  boxes_names.add(box_with_extension[0:box_with_extension.find('.mp4')])

sorted_boxes_names = list(boxes_names)
sorted_boxes_names.sort()

for box_name in sorted_boxes_names:
  print(box_name)

merged_jobs_names = set()
for time_cut_files in time_cuts_files:
  merged_jobs_names.add(time_cut_files[0:time_cut_files.find('_time_cut')])

for merged_job_name in merged_jobs_names:
  print(merged_job_name)

merged_jobs_names_sorted = list(merged_jobs_names)
merged_jobs_names_sorted.sort()

commands = []
for job_name in merged_jobs_names_sorted:
  file_path = folder + '/' + job_name
  if len(sorted_boxes_names) == 0:
    cuts_regex = file_path + '_time_cut*'
    output_file = file_path + '.mp4'
    command = 'mencoder -oac copy -ovc copy ' + cuts_regex + ' -o ' + output_file
    commands.append(command)
  else:
    for box_name in sorted_boxes_names:
      cuts_regex = file_path + '_time_cut*' + box_name + '*'
      output_file = file_path + '_' + box_name + '.mp4'
      command = 'mencoder -oac copy -ovc copy ' + cuts_regex + ' -o ' + output_file
      commands.append(command)

for command in commands:
  print(command)

for command in commands:
  print('now running: ' + command)
  subprocess.run(command, shell=True)

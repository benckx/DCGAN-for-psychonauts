import subprocess
import sys
from os import listdir
from os.path import isfile, join

folder = None
if len(sys.argv) > 1:
  if len(sys.argv) > 1:
    params = sys.argv[0:]
    for idx, param in enumerate(params):
      if param == '--folder':
        folder = params[idx + 1]

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
  for box_name in sorted_boxes_names:
    commands.append('mencoder -oac copy -ovc copy ' + file_path + '_time_cut*' + box_name + '*' + ' -o ' + file_path + '_' + box_name + '.mp4')

for command in commands:
  print(command)

# for command in commands:
#   print('now running: ' + command)
#   subprocess.run(command, shell=True)

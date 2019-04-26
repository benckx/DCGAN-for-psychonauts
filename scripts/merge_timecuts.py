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

# for time_cut_files in time_cuts_files:
#   print(time_cut_files)

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

merged_files_names = set()
for time_cut_files in time_cuts_files:
    merged_files_names.add(time_cut_files[0:time_cut_files.find('_time_cut')])

for merged_file_name in merged_files_names:
  print(merged_file_name)

merged_files_named_sorted = list(merged_files_names)
merged_files_named_sorted.sort()

commands = []
for element in merged_files_named_sorted:
  file_path = folder + '/' + element
  for box_name in sorted_boxes_names:
    commands.append('mencoder -oac copy -ovc copy ' + file_path + '_time_cut*' + box_name + '*' + ' -o ' + file_path + '_' + box_name + '.mp4')

for command in commands:
  print(command)

# for command in commands:
#   print('now running: ' + command)
#   subprocess.run(command, shell=True)

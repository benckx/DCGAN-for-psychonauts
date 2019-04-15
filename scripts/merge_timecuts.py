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

time_cuts = [f for f in listdir(folder) if (isfile(join(folder, f)) and f.find('_time_cut') > -1)]
time_cuts.sort()

merged_files_names = set()
for time_cut in time_cuts:
  merged_files_names.add(time_cut[0:time_cut.find('_time_cut')])

merged_files_named_sorted = list(merged_files_names)
merged_files_named_sorted.sort()

commands = []
for element in merged_files_named_sorted:
  file_path = folder + '/' + element
  commands.append('mencoder -oac copy -ovc copy ' + file_path + '_time_cut* -o ' + file_path + '.mp4')

for command in commands:
  print(command)

for command in commands:
  print('now running: ' + command)
  subprocess.run(command, shell=True)

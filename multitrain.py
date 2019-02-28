import subprocess
from os import listdir
from os.path import isfile, join

import pandas as pd

data_folders = [f for f in listdir('data/')]
csv_files = [f for f in listdir('.') if (isfile(join('.', f)) and f.endswith(".csv"))]

if len(csv_files) != 1:
  print('csv issues: ' + str(csv_files))
  exit(1)

data = pd.read_csv(csv_files[0], encoding='UTF-8')

for index, row in data.iterrows():
  if row['dataset'] not in data_folders:
    print('Error: dataset ' + row['dataset'] + ' not found !')
    exit(1)

for index, row in data.iterrows():
  print(str(row))

  # noinspection PyListCreation
  args = ["python3", "main.py"]

  args.append("--epoch")
  args.append(str(row['epoch']))

  args.append("--name")
  args.append(str(row['name']))

  args.append("--dataset")
  args.append(str(row['dataset']))

  args.append("--grid_width")
  args.append(str(row['grid_width']))
  args.append("--grid_height")
  args.append(str(row['grid_height']))

  args.append("--nbr_of_layers_g")
  args.append(str(row['nbr_of_layers_g']))
  args.append("--nbr_of_layers_d")
  args.append(str(row['nbr_of_layers_d']))

  if row['batch_norm_d']:
    args.append("--batch_norm_d")

  if row['batch_norm_g']:
    args.append("--batch_norm_g")

  args.append("--sample_rate")
  args.append("1")

  args.append("--train")

  subprocess.run(args)
else:
  print('Config file not found')

import os
import subprocess

import pandas as pd

if os.path.exists('config.csv'):
  data = pd.read_csv('config.csv', encoding='UTF-8')
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

    args.append("--use_checkpoints")
    args.append("False")

    args.append("--sample_rate")
    args.append("1")

    args.append("--train")

    subprocess.run(args)
else:
  print('Config file not found')

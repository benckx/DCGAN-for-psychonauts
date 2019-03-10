# noinspection PyListCreation
def build_dcgan_cmd(cmd_row):
  dcgan_cmd = ["python3", "main.py"]

  dcgan_cmd.append("--epoch")
  dcgan_cmd.append(str(cmd_row['epoch']))

  dcgan_cmd.append("--name")
  dcgan_cmd.append(cmd_row['name'])

  dcgan_cmd.append("--dataset")
  dcgan_cmd.append(cmd_row['dataset'])

  dcgan_cmd.append("--grid_width")
  dcgan_cmd.append(str(cmd_row['grid_width']))
  dcgan_cmd.append("--grid_height")
  dcgan_cmd.append(str(cmd_row['grid_height']))

  if cmd_row['nbr_of_layers_g']:
    dcgan_cmd.append("--nbr_of_layers_g")
    dcgan_cmd.append(str(cmd_row['nbr_of_layers_g']))

  if cmd_row['nbr_of_layers_d']:
    dcgan_cmd.append("--nbr_of_layers_d")
    dcgan_cmd.append(str(cmd_row['nbr_of_layers_d']))

  if cmd_row['batch_norm_g'] == '' or cmd_row['batch_norm_g']:
    dcgan_cmd.append("--batch_norm_g")

  if cmd_row['batch_norm_g'] == '' or cmd_row['batch_norm_d']:
    dcgan_cmd.append("--batch_norm_d")

  if cmd_row['activation_g'] and str(cmd_row['activation_g']) != "nan":
    dcgan_cmd.append("--activation_g")
    dcgan_cmd.append(cmd_row['activation_g'])

  if cmd_row['activation_d'] and str(cmd_row['activation_d']) != "nan":
    dcgan_cmd.append("--activation_d")
    dcgan_cmd.append(cmd_row['activation_d'])

  if cmd_row['learning_rate'] and not math.isnan(cmd_row['learning_rate']):
    dcgan_cmd.append("--learning_rate")
    dcgan_cmd.append(str(cmd_row['learning_rate']))

  if cmd_row['beta1'] and not math.isnan(cmd_row['beta1']):
    dcgan_cmd.append("--beta1")
    dcgan_cmd.append(str(cmd_row['beta1']))

  if cmd_row['nbr_g_updates'] and not math.isnan(cmd_row['nbr_g_updates']):
    dcgan_cmd.append('--nbr_g_updates')
    dcgan_cmd.append(str(int(cmd_row['nbr_g_updates'])))

  if cmd_row['use_checkpoints']:
    dcgan_cmd.append("--use_checkpoints")

  dcgan_cmd.append('--sample_rate')
  dcgan_cmd.append('1')

  dcgan_cmd.append('--train')

  return dcgan_cmd

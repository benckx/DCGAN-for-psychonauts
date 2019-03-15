import configparser
import datetime
import ftplib
import os
import threading
import time
import zipfile


def upload_via_ftp(file_name):
  try:
    config = configparser.ConfigParser()
    config.read('ftp.ini')
    session = ftplib.FTP(config['ftp']['host'], config['ftp']['user'], config['ftp']['password'])
    file = open(file_name, 'rb')
    session.storbinary('STOR ' + file_name, file)
    file.close()
    session.quit()
  except Exception as exception:
    print('error during FTP transfer -> {}'.format(exception))


def zip_folder(folder, zip_file_name):
  zipf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
  for root, dirs, files in os.walk(folder):
    for file in files:
      zipf.write(os.path.join(root, file))
  zipf.close()


def auto_save_checkpoint_scheduled(first_exec, checkpoint_name):
  print("auto_save_checkpoint_scheduled")

  if not first_exec:
    checkpoint_dir = 'checkpoint/' + checkpoint_name
    if os.path.exists(checkpoint_dir):
      ts = time.time()
      timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M')
      zip_file_name = checkpoint_name + "_" + timestamp + ".zip"
      zip_folder('checkpoint/' + checkpoint_name, zip_file_name)
      print('created zip: {}'.format(zip_file_name))
    else:
      print("{} doesn't exist yet".format(checkpoint_dir))
  else:
    print("first exec, do nothing, wait some time")

  threading.Timer(360.0, auto_save_checkpoint_scheduled, args=[False, checkpoint_name]).start()

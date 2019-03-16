import configparser
import datetime
import ftplib
import os
import os.path
import os.path
import time
import zipfile
from multiprocessing import Pool

ftp_threads_pool = Pool(processes=5)


def upload_via_ftp(file_name):
  ftp_threads_pool.apply(upload_via_ftp_sync, args=[file_name])


def upload_via_ftp_sync(file_name):
  print('uploading {} to ftp'.format(file_name))
  try:
    if os.path.isfile('ftp.ini'):
      config = configparser.ConfigParser()
      config.read('ftp.ini')
      session = ftplib.FTP(config['ftp']['host'], config['ftp']['user'], config['ftp']['password'])
      file = open(file_name, 'rb')
      session.storbinary('STOR ' + file_name, file)
      file.close()
      session.quit()
    else:
      print('Warn: ftp.ini not found, file {} can not be sent to ftp'.format(file_name))
  except Exception as exception:
    print('Error during FTP transfer -> {}'.format(exception))


def zip_folder(folder, zip_file_name):
  zipf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
  for root, dirs, files in os.walk(folder):
    for file in files:
      zipf.write(os.path.join(root, file))
  zipf.close()


def save_checkpoint(checkpoint_name):
  print("auto_save_checkpoint_scheduled")
  checkpoint_dir = 'checkpoint/' + checkpoint_name
  if os.path.exists(checkpoint_dir):
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M')
    zip_file_name = checkpoint_name + "_" + timestamp + ".zip"
    zip_folder('checkpoint/' + checkpoint_name, zip_file_name)
    print('created zip: {}'.format(zip_file_name))
    # upload_via_ftp(zip_file_name)
  else:
    print("{} doesn't exist yet".format(checkpoint_dir))

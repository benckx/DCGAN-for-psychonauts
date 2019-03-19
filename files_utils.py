import configparser
import datetime
import ftplib
import os
import os.path
import os.path
import time
import zipfile
from multiprocessing import Pool

ftp_threads_pool = Pool(processes=10)


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
      print('{} correctly uploaded to ftp'.format(file_name))
    else:
      print('Warn: ftp.ini not found, file {} can not be sent to ftp'.format(file_name))
  except Exception as e:
    print('Error during FTP transfer of file {} -> {}'.format(file_name, e))


def upload_via_ftp(file_name):
  try:
    print('Before adding {} to ftp thread pool'.format(file_name))
    ftp_threads_pool.apply_async(upload_via_ftp_sync, args=[file_name])
  except Exception as e:
    print('Error during FTP thread pool queuing of file {} -> {}'.format(file_name, e))


def zip_folder(folder, zip_file_name):
  zipf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
  for root, dirs, files in os.walk(folder):
    for file in files:
      zipf.write(os.path.join(root, file))
  zipf.close()


def backup_checkpoint(checkpoint_name):
  try:
    print('backing up checkpoint...')
    checkpoint_dir = 'checkpoint/' + checkpoint_name
    if os.path.exists(checkpoint_dir):
      ts = time.time()
      timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M')
      zip_file_name = checkpoint_name + "_" + timestamp + ".zip"
      zip_folder('checkpoint/' + checkpoint_name, zip_file_name)
      print('created zip: {}'.format(zip_file_name))
      upload_via_ftp(zip_file_name)
    else:
      print('{} does not exist'.format(checkpoint_dir))
  except Exception as e:
    print('Error during checkpoint backup: {}'.format(e))


def must_backup_checkpoint():
  try:
    if os.path.isfile('ftp.ini'):
      config = configparser.ConfigParser()
      config.read('ftp.ini')
      return config['checkpoint']['backup']
  except Exception as e:
    print('Error: {}'.format(e))

  return False


def get_checkpoint_backup_delay():
  """ In minutes """
  try:
    if os.path.isfile('ftp.ini'):
      config = configparser.ConfigParser()
      config.read('ftp.ini')
      return int(config['checkpoint']['backup_delay'])
  except Exception as e:
    print('Error: {}'.format(e))

  return 4 * 60

from tensorflow.python.client import device_lib


class GpuIterator:
  def __init__(self):
    self.idx = 1
    self.devices = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    for device in self.devices:
      print(str(device))

  def next(self):
    selected_device = self.devices[self.idx]
    if self.idx < len(self.devices) - 1:
      self.idx += 1
    else:
      self.idx = 1

    print('assigning load to {}'.format(selected_device.name))
    return selected_device.name
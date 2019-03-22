from tensorflow.python.client import device_lib


class GpuIterator:
  def __init__(self):
    self.idx = 0
    self.devices = device_lib.list_local_devices()
    for device in self.devices:
      print(str(device))

  def next(self):
    selected_device = self.devices[self.idx]
    if self.idx < len(self.devices) - 1:
      self.idx += 1
    else:
      self.idx = 0

    print('assigning load to {}'.format(selected_device.name))
    return selected_device.name

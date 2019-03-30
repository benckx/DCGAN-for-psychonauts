from tensorflow.python.client import device_lib


class GpuAllocator:
  def __init__(self, gpu_idx):
    self.idx = 1
    self.devices = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    self.nbr_devices = len(self.devices)
    if gpu_idx is not None:
      self.gpu_idx = int(gpu_idx)
    else:
      self.gpu_idx = None
    for device in self.devices:
      print(str(device.name))

  def generator_device(self):
    if self.gpu_idx is not None:
      return self.devices[self.gpu_idx].name
    elif self.nbr_devices == 5:
      return self.devices[4].name
    elif self.nbr_devices == 4:
      return self.devices[3].name
    elif self.nbr_devices == 2:
      return self.devices[1].name

  def sampler_device(self):
    if self.gpu_idx is not None:
      return self.devices[self.gpu_idx].name
    elif self.nbr_devices == 5:
      return self.devices[3].name
    elif self.nbr_devices == 4:
      return self.devices[2].name
    elif self.nbr_devices == 2:
      return self.devices[1].name

  def discriminator_device(self):
    if self.gpu_idx is not None:
      return self.devices[self.gpu_idx].name
    elif self.nbr_devices == 5:
      return self.devices[2].name
    elif self.nbr_devices == 4:
      return self.devices[1].name
    elif self.nbr_devices == 2:
      return self.devices[0].name

  def discriminator_fake_device(self):
    if self.gpu_idx is not None:
      return self.devices[self.gpu_idx].name
    elif self.nbr_devices == 5:
      return self.devices[1].name
    elif self.nbr_devices == 4:
      return self.devices[0].name
    elif self.nbr_devices == 2:
      return self.devices[0].name

  def other_things_device(self):
    if self.gpu_idx is not None:
      return self.devices[self.gpu_idx].name
    else:
      return self.devices[len(self.devices) - 1]

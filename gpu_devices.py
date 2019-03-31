from tensorflow.python.client import device_lib


class GpuAllocator:
  def __init__(self, gpu_idx):
    self.idx = 1
    self.gpu_devices = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    self.cpu_devices = [x for x in device_lib.list_local_devices() if x.device_type == 'CPU']
    self.nbr_gpu_devices = len(self.gpu_devices)
    if gpu_idx is not None:
      self.gpu_idx = int(gpu_idx)
    else:
      self.gpu_idx = None
    for gpu_devices in self.gpu_devices:
      print(str(gpu_devices.name))
    for cpu_devices in self.cpu_devices:
      print(str(cpu_devices.name))

  def generator_device(self):
    if len(self.gpu_devices) == 0:
      return ''
    elif self.gpu_idx is not None:
      return self.gpu_devices[self.gpu_idx].name
    elif self.nbr_gpu_devices == 5:
      return self.gpu_devices[4].name
    elif self.nbr_gpu_devices == 4:
      return self.gpu_devices[3].name
    elif self.nbr_gpu_devices == 2:
      return self.gpu_devices[1].name

  def sampler_device(self):
    if len(self.gpu_devices) == 0:
      return ''
    elif self.gpu_idx is not None:
      return self.gpu_devices[self.gpu_idx].name
    elif self.nbr_gpu_devices == 5:
      return self.gpu_devices[3].name
    elif self.nbr_gpu_devices == 4:
      return self.gpu_devices[2].name
    elif self.nbr_gpu_devices == 2:
      return self.gpu_devices[1].name

  def discriminator_device(self):
    if len(self.gpu_devices) == 0:
      return ''
    elif self.gpu_idx is not None:
      return self.gpu_devices[self.gpu_idx].name
    elif self.nbr_gpu_devices == 5:
      return self.gpu_devices[2].name
    elif self.nbr_gpu_devices == 4:
      return self.gpu_devices[1].name
    elif self.nbr_gpu_devices == 2:
      return self.gpu_devices[0].name

  def discriminator_fake_device(self):
    if len(self.gpu_devices) == 0:
      return ''
    elif self.gpu_idx is not None:
      return self.gpu_devices[self.gpu_idx].name
    elif self.nbr_gpu_devices == 5:
      return self.gpu_devices[1].name
    elif self.nbr_gpu_devices == 4:
      return self.gpu_devices[0].name
    elif self.nbr_gpu_devices == 2:
      return self.gpu_devices[0].name

  def other_things_device(self):
    if len(self.gpu_devices) == 0:
      return ''
    elif self.gpu_idx is not None:
      return self.gpu_devices[self.gpu_idx].name
    else:
      return self.gpu_devices[len(self.gpu_devices) - 1]

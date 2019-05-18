from tensorflow.python.client import device_lib


class GpuAllocator:
  def __init__(self, gpu_idx):
    if gpu_idx is not None:
      self.gpu_devices = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
      self.cpu_devices = [x for x in device_lib.list_local_devices() if x.device_type == 'CPU']
      self.nbr_gpu_devices = len(self.gpu_devices)
      self.gpu_idx = int(gpu_idx)
      for gpu_devices in self.gpu_devices:
        print(str(gpu_devices.name))
      for cpu_devices in self.cpu_devices:
        print(str(cpu_devices.name))
    else:
      self.gpu_idx = None

    self.allocations = {}
    if self.gpu_idx is not None:
      device = '/device:GPU:{}'.format(gpu_idx)
      self.allocations['generator'] = device
      self.allocations['discriminator'] = device
      self.allocations['sampler'] = device
      self.allocations['discriminator_fake'] = device
      self.allocations['other'] = device
    elif self.nbr_gpu_devices == 2:
      self.allocations['generator'] = self.gpu_devices[self.nbr_gpu_devices - 1].name
      self.allocations['sampler'] = self.gpu_devices[self.nbr_gpu_devices - 1].name
      self.allocations['discriminator'] = self.gpu_devices[self.nbr_gpu_devices - 2].name
      self.allocations['discriminator_fake'] = self.gpu_devices[self.nbr_gpu_devices - 2].name
      self.allocations['other'] = self.gpu_devices[self.nbr_gpu_devices - 2].name
    elif self.nbr_gpu_devices >= 3:
      self.allocations['generator'] = self.gpu_devices[self.nbr_gpu_devices - 1].name
      self.allocations['discriminator'] = self.gpu_devices[self.nbr_gpu_devices - 2].name
      self.allocations['sampler'] = self.gpu_devices[self.nbr_gpu_devices - 3].name
      self.allocations['discriminator_fake'] = self.gpu_devices[self.nbr_gpu_devices - 3].name
      self.allocations['other'] = self.gpu_devices[self.nbr_gpu_devices - 3].name
    else:
      self.allocations['generator'] = None
      self.allocations['discriminator'] = None
      self.allocations['sampler'] = None
      self.allocations['discriminator_fake'] = None
      self.allocations['other'] = None

  def generator_device(self):
    if self.allocations['generator'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['generator']]

  def sampler_device(self):
    if self.allocations['sampler'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['sampler']]

  def discriminator_device(self):
    if self.allocations['discriminator'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['discriminator']]

  def discriminator_fake_device(self):
    if self.allocations['discriminator_fake'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['discriminator_fake']]

  def other_things_device(self):
    if self.allocations['other'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['other']]

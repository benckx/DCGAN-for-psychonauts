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

    self.allocations = {}
    if gpu_idx is not None:
      self.allocations['generator'] = gpu_idx
      self.allocations['discriminator'] = gpu_idx
      self.allocations['sampler'] = gpu_idx
      self.allocations['discriminator_fake'] = gpu_idx
      self.allocations['other'] = gpu_idx
    elif self.nbr_gpu_devices == 2:
      self.allocations['generator'] = self.nbr_gpu_devices - 1
      self.allocations['sampler'] = self.nbr_gpu_devices - 1
      self.allocations['discriminator'] = self.nbr_gpu_devices - 2
      self.allocations['discriminator_fake'] = self.nbr_gpu_devices - 2
      self.allocations['other'] = self.nbr_gpu_devices - 2
    elif self.nbr_gpu_devices >= 3:
      self.allocations['generator'] = self.nbr_gpu_devices - 1
      self.allocations['discriminator'] = self.nbr_gpu_devices - 2
      self.allocations['sampler'] = self.nbr_gpu_devices - 3
      self.allocations['discriminator_fake'] = self.nbr_gpu_devices - 3
      self.allocations['other'] = self.nbr_gpu_devices - 3
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
      return self.gpu_devices[self.allocations['generator']].name

  def sampler_device(self):
    if self.allocations['sampler'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['sampler']].name

  def discriminator_device(self):
    if self.allocations['discriminator'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['discriminator']].name

  def discriminator_fake_device(self):
    if self.allocations['discriminator_fake'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['discriminator_fake']].name

  def other_things_device(self):
    if self.allocations['other'] is None:
      return ''
    else:
      return self.gpu_devices[self.allocations['other']].name

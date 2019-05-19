from tensorflow.python.client import device_lib


class GpuAllocator:
  def __init__(self, gpu_idx):
    if gpu_idx is None:
      self.gpu_idx = None
      self.gpu_devices = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
      self.cpu_devices = [x for x in device_lib.list_local_devices() if x.device_type == 'CPU']
      self.nbr_gpu_devices = len(self.gpu_devices)
      for gpu_devices in self.gpu_devices:
        print(str(gpu_devices.name))
      for cpu_devices in self.cpu_devices:
        print(str(cpu_devices.name))
    else:
      self.nbr_gpu_devices = 1
      self.gpu_idx = int(gpu_idx)

    self.allocations = {}
    if self.gpu_idx is not None:
      device = '/device:GPU:{}'.format(self.gpu_idx)
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
      self.allocations['generator'] = ''
      self.allocations['discriminator'] = ''
      self.allocations['sampler'] = ''
      self.allocations['discriminator_fake'] = ''
      self.allocations['other'] = ''

  def generator_device(self):
    return self.allocations['generator']

  def sampler_device(self):
    return self.allocations['sampler']

  def discriminator_device(self):
    return self.allocations['discriminator']

  def discriminator_fake_device(self):
    return self.allocations['discriminator_fake']

  def other_things_device(self):
    return self.allocations['other']

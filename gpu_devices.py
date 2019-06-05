from tensorflow.python.client import device_lib


class GpuAllocator:
  def __init__(self):
    gpu_devices = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    nbr_gpu_devices = len(gpu_devices)
    for gpu_devices in gpu_devices:
      print(str(gpu_devices.name))

    self.allocations = {}
    if nbr_gpu_devices == 1:
      self.allocations['generator'] = gpu_devices[0].name
      self.allocations['discriminator'] = gpu_devices[0].name
      self.allocations['sampler'] = gpu_devices[0].name
      self.allocations['discriminator_fake'] = gpu_devices[0].name
      self.allocations['other'] = gpu_devices[0].name
    elif nbr_gpu_devices == 2:
      self.allocations['generator'] = gpu_devices[0].name
      self.allocations['sampler'] = gpu_devices[0].name
      self.allocations['discriminator'] = gpu_devices[1].name
      self.allocations['discriminator_fake'] = gpu_devices[1].name
      self.allocations['other'] = gpu_devices[1].name
    elif nbr_gpu_devices >= 3:
      self.allocations['generator'] = gpu_devices[nbr_gpu_devices - 1].name
      self.allocations['discriminator'] = gpu_devices[nbr_gpu_devices - 2].name
      self.allocations['sampler'] = gpu_devices[nbr_gpu_devices - 3].name
      self.allocations['discriminator_fake'] = gpu_devices[nbr_gpu_devices - 3].name
      self.allocations['other'] = gpu_devices[nbr_gpu_devices - 3].name
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

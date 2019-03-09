class ThreadsSharedState:
  def __init__(self):
    self.sample_folder = None
    self.job_name = None
    self.nbr_of_frames = None
    self.current_cut = None
    self.sample_res = None
    self.render_res = None

  def get_folder(self):
    return self.sample_folder

  def set_folder(self, folder):
    self.sample_folder = folder

  def get_job_name(self):
    return self.job_name

  def set_job_name(self, job_name):
    self.job_name = job_name

  def get_nbr_of_frames(self):
    return self.nbr_of_frames

  def set_nbr_of_frames(self, nbr_frames):
    self.nbr_of_frames = nbr_frames

  def get_current_cut(self):
    return self.current_cut

  def increment_cut(self):
    self.current_cut += 1

  def get_sample_res(self):
    return self.sample_res

  def set_sample_res(self, res):
    self.sample_res = res

  def get_render_res(self):
    return self.render_res

  def set_render_res(self, res):
    self.render_res = res

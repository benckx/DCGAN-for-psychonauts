class ThreadsSharedState:
  def __init__(self):
    self.sample_folder = None
    self.job_name = None
    self.frames_threshold = None
    self.current_cut = 1
    self.sample_res = None
    self.render_res = None
    self.upload_to_ftp = False
    self.delete_at_the_end = False

  def get_folder(self):
    return self.sample_folder

  def set_folder(self, folder):
    self.sample_folder = folder

  def get_job_name(self):
    return self.job_name

  def set_job_name(self, job_name):
    self.job_name = job_name

  def get_frames_threshold(self):
    return self.frames_threshold

  def set_frames_threshold(self, nbr_frames):
    self.frames_threshold = nbr_frames

  def get_current_cut(self):
    return self.current_cut

  def set_current_cut(self, cut):
    self.current_cut = cut

  def init_current_cut(self):
    self.current_cut = 1

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

  def is_upload_to_ftp(self):
    return self.upload_to_ftp

  def set_upload_to_ftp(self, upload_to_ftp):
    self.upload_to_ftp = upload_to_ftp

  def is_delete_at_the_end(self):
    return self.delete_at_the_end

  def set_delete_at_the_end(self, upload_to_ftp):
    self.delete_at_the_end = upload_to_ftp

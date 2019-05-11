def min_to_string(minutes):
  if minutes <= 120:
    return '{:0.2f} minutes'.format(minutes)
  else:
    hours = minutes / 60
    return '{:0.2f} hours'.format(hours)

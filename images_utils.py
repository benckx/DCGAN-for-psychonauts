def get_boxes(sample_res, render_res):
  if sample_res[0] % render_res[0] != 0 or sample_res[1] % render_res[1] != 0:
    print('Error: Resolution not divisible: {}, {}'.format(sample_res, render_res))
    exit(1)

  boxes = []
  x_cuts = int(sample_res[0] / render_res[0])
  y_cuts = int(sample_res[1] / render_res[1])
  for x in range(0, x_cuts):
    for y in range(0, y_cuts):
      x1 = x * render_res[0]
      y1 = y * render_res[1]
      x2 = (x + 1) * render_res[0]
      y2 = (y + 1) * render_res[1]
      boxes.append([x1, y1, x2, y2])

  return boxes

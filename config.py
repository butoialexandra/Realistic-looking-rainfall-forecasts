import numpy as np

input_shape = (127, 188)
output_shape = (295, 427)

# TODO: make sure we didn't mix up x, y from dataset
in_pixels = int(np.prod(input_shape))
out_pixels = int(np.prod(output_shape))

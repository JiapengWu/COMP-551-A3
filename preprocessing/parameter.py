from scipy import ndimage
thresh_hold = 225
blur_function = [ndimage.maximum_filter, ndimage.median_filter]
filename = ["maximum", "median"]
blur_parameter = 2

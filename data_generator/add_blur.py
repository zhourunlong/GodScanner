import os
import sys
import random
from PIL import Image, ImageFilter

input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

figures = os.listdir(input_path)

for figure in figures:
    im1 = Image.open(os.path.join(input_path, figure))
    method = random.random()
    if method > 1.:
        radius = random.random() * 1. + 2.5
        im2 = im1.filter(ImageFilter.GaussianBlur(radius = radius)) 
    else:
        radius = random.random() * 1. + 2.
        im2 = im1.filter(ImageFilter.BoxBlur(radius = radius)) 
    im2.save(os.path.join(output_path, figure), format='png')
    print(figure + ' finished. ')
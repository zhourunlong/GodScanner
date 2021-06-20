import os
import sys
import random

input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')

figures = os.listdir(input_path)

print(figures)

def Unicode():
    val = random.randint(0x4e00, 0x9fbf)
    # return chr(0x263a)
    return 'I' + str(chr(9829)) + 'DeepLearning'

for figure in figures:
    input_file = os.path.join(input_path, figure)
    angle = random.randint(1, 180)
    color = '#808080'
    size = random.randint(50, 100)
    opacity = random.random() * 0.3 + 0.15
    space = random.randint(100, 200)
    print("python marker.py -f {} -m {} --size {} --angle {} -s {} --opacity {} -c {}".format(input_file, Unicode(), size, angle, space, opacity, color))
    os.system("python marker.py -f {} -m {} --size {} --angle {} -s {} --opacity {} -c {}".format(input_file, Unicode(), size, angle, space, opacity, color))
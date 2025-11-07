import numpy as np
from lxml import etree
from pathlib import Path

current_file = Path(__file__).resolve()
ROOT = str(current_file.parent)
tree = etree.parse(ROOT + "/scene.xml")
root = tree.getroot()
world = root.find("worldbody")

num_boxes = 10
cube_size_base = 0.15
x_range=np.linspace(1.,8., num_boxes)
y_range=(-0.2, 0.2)

for i in range(num_boxes):
    x = x_range[i] + np.random.normal(0.0, 0.1)
    y = np.random.uniform(*y_range)
    cube_dim = np.random.uniform(0.1, 0.4)
    body = etree.SubElement(world, "body", name=f"cube_body{i}", pos=f"{x} {y} {.5*cube_dim}")
    
    # etree.SubElement(body, "freejoint")
    # etree.SubElement(body, "joint", 
    #                 name=f"cube_slide_x{i}",
    #                 type="slide",
    #                 axis="1 0 0",
    #                 damping="0.001",
    #                 range="-1 1")
    # etree.SubElement(body, "joint",
    #                 name=f"cube_slide_y{i}",
    #                 type="slide",
    #                 axis="0 1 0",
    #                 damping="0.001",
    #                 range="-1 1")
    etree.SubElement(body, "geom",
                     name=f"cube{i}",
                     type="box",
                     size=f"{cube_dim} {cube_dim} {cube_dim}",
                     rgba="0.8 0.2 0.2 1",
                     contype="1",
                     conaffinity="1")

tree.write(ROOT + "/walker_scene_with_random_boxes.xml", pretty_print=True, xml_declaration=True, encoding="UTF-8")


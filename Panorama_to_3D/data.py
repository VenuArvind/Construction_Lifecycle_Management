import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt 

path_1 = "C:\\Users\\kumaran\\Desktop\\fyp_dataset\\data\\val_PC\\val\\PCval_0001.pkl"

path_2 = "C:\\Users\\kumaran\\Desktop\\fyp_dataset\\data\\train_stanford\\area1_0000.pkl"

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi

def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = coorx2u(corU[:, 0])
    vU = coory2v(corU[:, 1])
    vB = coory2v(corB[:, 1])

    x, y = uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)

def pixel_to_spherical(pixel_coords, image_width, image_height):
    # Convert pixel coordinates to spherical coordinates
    u = (pixel_coords[0] / image_width) * 2 * np.pi
    v = ((pixel_coords[1] / image_height) - 0.5) * np.pi
    return u, v

def spherical_to_cartesian(spherical_coords, radius):
    # Convert spherical coordinates to 3D Cartesian coordinates
    u, v = spherical_coords
    x = radius * np.sin(v) * np.cos(u)
    y = radius * np.sin(v) * np.sin(u)
    z = radius * np.cos(v)
    return x, y, z



with open(path_1, 'rb') as f:
    data = pickle.load(f)
    print(data["cor"])

    image = data["image"]

    cor = data["cor"].astype(np.uint8)
    
    



    

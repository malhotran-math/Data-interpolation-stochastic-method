import math
import numpy as np
from scipy.interpolate import CubicSpline
# import read_data

# xf, yf, vf = read_data.read_field("high-res-bathymetry")

"""There are two ways of representing the data, either as a grid, consisting of three arrays: x, y, v.
    Here x and y are arrays containing the possible x-coordinates and y-coordinates of grid points, and
    v[i][j] is the height at the point (x[i], y[j]).
    
    The variable data_set will be a list of all points (x, y, z) with z the depth at the point (x, y).
"""


def generate_data_set(x, y, v):
    """Given a gridded input with height v[i][j] at the point (x[i], y[j]), return an array of all x, y, z pairs"""
    data = []
    for i in range(len(x)):
        for j in range(len(y)):
            if not np.isnan(v[i][j]):
                data.append([x[i], y[j], v[i][j]])
    return np.array(data)


def centerpoint(data_set):
    """ Computes the center of the dataset """
    return sum(data_set / len(data_set))


def uniform_rays(number_of_rays):
    """ Computes the angles for number_of_rays rays """
    ray_angles = [2*np.pi*k/number_of_rays for k in number_of_rays]
    return ray_angles

def data_in_cone(data_set, centerpoint, ray_angle, cone_angle):
    """ returns the datapoints which are closest to a given ray """
    data_cone = []
    for v in data_set:
        if np.abs(np.angle((v[0], v[1])) - ray_angle) <= cone_angle:
            data_cone.append(v)
    return data_cone


def interpolate_grid(point, xf, yf, vf):
    """ Compute the interpolated value at a point, taking as input the data (xf, yf, vf) and the point at which
        we want to know the interpolated value (x0, y0)
    """
    vf_modified = np.where(np.isnan(vf), 0, vf)

    cxf = CubicSpline(xf, vf_modified, bc_type='clamped', extrapolate=False)
    vinterf = cxf(point[0])

    cyf = CubicSpline(yf, vinterf, axis=1, bc_type = 'clamped', extrapolate=False)
    vc2f = cyf(point[1])
    return vc2f


def points_on_ray(centerpoint, ray_angle, distances_from_centerpoint):
    """ Returns (x,y) coordinates of points on the ray whose distances from the center is given by disctances_from_centerpoint"""
    return [centerpoint + np.vector(r*np.exp(1j*ray_angle)) for r in distances_from_centerpoint]


def ray_interpolate_height(centerpoint, ray_angle, cone_angle, distances_from_centerpoint, data_set, xf, yf, vf):
    """ Returns interpolated height of the points on the ray """
    coordinates = points_on_ray(centerpoint, ray_angle, distances_from_centerpoint)
    close_points = data_in_cone(data_set, centerpoint, ray_angle, cone_angle)
    return [(v[0], v[1], interpolate_grid(v, xf, yf, vf)) for v in coordinates]

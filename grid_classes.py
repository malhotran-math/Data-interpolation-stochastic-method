import math
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interpn
import basic_functions
import matplotlib.pyplot as plt


""" The angle theta will always be in radians"""


def polar_to_rect(theta, r, cx, cy):
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return [x, y]


def rect_to_polar(x, y, cx, cy):
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.asin((y - cy) / r)
    return [r, theta]


def rect_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def polar_dist(r1, theta1, r2, theta2):
    """Compute distance between two polar points"""
    return rect_dist(r1 * math.cos(theta1), r1 * math.sin(theta1), r2 * math.cos(theta2), r2 * math.sin(theta2))


def find_index(lst, val):
    """Given a sorted list lst and a value val, find the index i with the property that lst[i] <= val <= lst[i+1]"""
    '''lst2 = np.array(lst)
    if val < lst[0]:
        return 0
    n = len(lst)
    if val > lst[-1]:
        return n - 2
    if n <= 1:
        return 0 
    closest = min(lst, key=lambda x:abs(x-val))
    if closest <=val:
        return np.where(lst2 == closest) #lst.index(closest)
    else:
        return np.where(lst2 == closest) - 1'''
    if val < lst[0]:
        return 0
    n = len(lst)
    if val > lst[-1]:
        return n - 2
    if n <= 1:
        return 0
    b = 0      # begin
    e = n - 1  # end
    i = n // 2
    while val < lst[i] or val > lst[i + 1]:
        if val < lst[i]:
            e = i
        else:
            b = i
        i = b + (e - b) // 2
        if i == n - 1:
            return i - 1
    return i


class RectangularGrid:
    def __init__(self, xs, ys, v):
        self.xs = xs
        self.ys = ys
        self.v = v

    def linear_interpolate(self, x, y):
        i = find_index(self.xs, x)
        j = find_index(self.ys, y)
        points = [[self.xs[i], self.ys[j]], [self.xs[i+1], self.ys[j]], [self.xs[i], self.ys[j+1]], [self.xs[i+1], self.ys[j+1]]]
        values = [self.v[i][j], self.v[i+1][j], self.v[i][j+1], self.v[i+1][j+1]]
        for v in values:
            if np.isnan(v):
                return np.nan
        L = LinearNDInterpolator(points, values)
        return L(x, y)

    def contains_point(self, x, y):
        """ Checks if the point (x,y) is contained within the rectangular grid """
        if self.xs[0] <= x <= self.xs[-1]:
            return self.ys[0] <= y <= self.ys[-1]

    def find_closest_point_value(self, x, y):
        """ Given the point (x,y), find the value of the closest point of the rectangular grid """
        i = find_index(self.xs, x)
        j = find_index(self.ys, y)
        return self.v[i][j]


    def plot(self):
        plt.contourf(self.xs, self.ys, np.transpose(np.array(self.v)))
        plt.show()


class PolarGrid:
    def __init__(self, thetas, rs, v, center_x, center_y):
        self.thetas = thetas
        self.rs = rs
        self.v = v
        self.center_x = center_x
        self.center_y = center_y


    def linear_interpolate(self, x, y):
        pol = rect_to_polar(x, y, self.center_x, self.center_y)
        i = find_index(self.thetas, pol[0])
        j = find_index(self.rs, pol[1])
        points = [polar_to_rect(self.thetas[i], self.rs[j], self.center_x, self.center_y),
                  polar_to_rect(self.thetas[i+1], self.rs[j], self.center_x, self.center_y),
                  polar_to_rect(self.thetas[i], self.rs[j+1], self.center_x, self.center_y),
                  polar_to_rect(self.thetas[i+1], self.rs[j+1], self.center_x, self.center_y)]
        values = [self.v[i][j], self.v[i+1][j], self.v[i][j+1], self.v[i+1][j+1]]
        for v in values:
            if np.isnan(v):
                return np.nan
        L = LinearNDInterpolator(points, values)
        return L(x, y)

    def plot(self, levels=20, cmap = 'Greys'):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        v = np.transpose(np.array(self.v))
        cax = ax.contourf(self.thetas, self.rs, v, levels=levels, cmap=cmap)
        plt.show()


class UnstructuredGrid:
    def __init__(self, array_of_points):
        self.points = array_of_points # self.sort_points(array_of_points)
        #for i in range(len(self.points)-1):
        #    point = self.points[i]
        #    next_point = self.points[i+1]
        #    if point[0] == next_point[0] and point[1] == next_point[1]:
        #        raise ValueError("Two points with same x,y coordinates!")


    #def sort_points(self, points):
    #    return points.sorted()

    def union(self, other):
        points_union = self.points + other.points
        return UnstructuredGrid(points_union)

    def weighted_average(self, other, weight_function):
        new_points = []
        for i in range(len(self.points-1)):
            point_s = self.points[i]
            point_o = other.points[i]
            if point_s[0] != point_o[0] or point_s[1] != point_o[1]:
                raise ValueError(" Not the same base grid")
            x = point_s[0]
            y = point_s[1]
            w = weight_function(x,y)
            z = point_s[2] * w + point_o[2] * (1 - w)
            new_points.append(np.array([x,y,z]))
        return UnstructuredGrid(new_points)

    def plot(self):
        xs = []
        ys = []
        z_vals = []
        for p in self.points:
            xs.append(p[0])
            ys.append(p[1])
            z_vals.append(p[2])
        plt.scatter(xs, ys, c=z_vals)
        plt.show()


def polar_grid_from_rectangular(R, n_thetas, dr, r_min, r_max, center=None):
    """ Generate a polar grid from a rectangular grid, with a specified angle between the rays """
    data = basic_functions.generate_data_set(R.xs, R.ys, R.v)
    if center is None:
        center = basic_functions.centerpoint(data)
    thetas = [2*np.pi*k/n_thetas for k in range(n_thetas)]
    num_of_r = math.floor((r_max - r_min) / dr)
    rs = [r_min + k * dr for k in range(num_of_r)]
    vs = []
    for i in range(len(thetas)):
        v_row = []
        for j in range(len(rs)):
            coord = polar_to_rect(thetas[i], rs[j], center[0], center[1])
            val = R.linear_interpolate(coord[0], coord[1])
            v_row.append(val)
        vs.append(v_row)
    return PolarGrid(thetas, rs, vs, center[0], center[1])


def polar_to_unsorted_grid(P):
    lst = []
    for i in range(len(P.thetas)):
        for j in range(len(P.rs)):
            coords = polar_to_rect(P.thetas[i], P.rs[j], P.center_x, P.center_y)
            lst.append([coords[0], coords[1], P.v[i][j]])
    return UnstructuredGrid(lst)


if __name__ == "__main__":
    n = 100
    grid = []
    for i in range(n):
        row = []
        for j in range(n):
            if n // 4 <= i <= 3 * n // 4 and n // 4 <= j <= 3 * n // 4:
                row.append(1)
            else:
                row.append(0)
        grid.append(row)

    data = np.array(grid)
    xs = np.linspace(0, 100, n)
    ys = np.linspace(0, 100, n)

    R = RectangularGrid(xs, ys, data)

    R.plot()

    nthetas = 60
    dr = 1
    r_min = 0
    r_max = 60

    P = polar_grid_from_rectangular(R, nthetas, dr, r_min, r_max)
    P.plot()

    UG = polar_to_unsorted_grid(P)
    UG.plot()




class RadialSlice:
    """This class contains function to polynomially extrapolate fine data and take the weighted average with coarse data
    input: A radial slice containing the following data:
    rs - a list of radius values used in the computation
    fine_values - a list of interpolated fine data values at the radius values, which cuts off at some point
    course_values - a list of interpolated course data values at all the radius values
    extrapolated_values (optional) - a list of extrapolated fine data values at all the radius values"""
    def __init__(self,
                 rs,
                 fine_values,
                 coarse_values,
                 extrapolated_values=None):
        self.rs = rs
        self.fine_values = fine_values
        self.coarse_values = coarse_values
        self.extrapolated_values = extrapolated_values
        self.averaged_values = None


    def polynomial_extrapolation(self, deg):
        """Extends the fine date using polynomial extrapolation"""
        n = len(self.fine_values)
        poly = np.polynomial.polynomial.Polynomial.fit(self.rs[:n], self.fine_values, deg).convert().coef
        new_values = [sum([poly[i]*x**i for i in range(deg+1)]) for x in self.rs[n:]]
        extrapolated_values = self.fine_values + new_values
        return extrapolated_values

    def compute_area(self):
        """Integrates the area of the distance between fine data and course data to compute confidence interval length"""
        fine_values = self.fine_values
        coarse_values = self.coarse_values
        rs = self.rs
        n = len(fine_values)
        delta = [np.abs(fine_values[i]-coarse_values[i]) for i in range(n)]
        return sum( [(delta[i]+delta[i+1])*(rs[i+1]-rs[i])/2 for i in range(n-1)])

    def weighted_average(self, local_gamma):
        """Computes the weighted average of extrapolated fine data and course data"""
        rs = self.rs
        coarse_values = self.coarse_values
        fine_values = self.fine_values
        n = len(fine_values)
        A = self.compute_area()
        r0 = rs[n-1]
        confidence_interval_length = min(local_gamma*A/n, rs[-1] - r0)
        extrapolated_values = self.extrapolated_values
        def weight(x):
            if x >=r0 and x <= r0+confidence_interval_length:
                return (np.cos(np.pi*(x - r0)/confidence_interval_length)+1)/2
            elif x <= r0:
                return 1
            else:
                return 0
        averaged_values = [ extrapolated_values[i]*weight(rs[i])+coarse_values[i]*(1-weight(rs[i]))
                            for i in range(len(rs))]
        return averaged_values

    def plot(self, title=None):
        """plots all the data"""
        n = len(self.fine_values)
        plot = plt.plot(self.rs[:n], self.fine_values, label="fine data")
        plot += plt.plot(self.rs, self.coarse_values, label="coarse data")
        plt.xlabel("Distance from midpoint [m]")
        plt.ylabel("Depth [m]")
        plt.title = title
        plt.legend()
        plt.show()

        n = len(self.fine_values)
        plot = plt.plot(self.rs[:n], self.fine_values, label="fine data")
        plot += plt.plot(self.rs, self.coarse_values, label="coarse data")
        plt.xlabel("Distance from midpoint [m]")
        plt.ylabel("Depth [m]")
        plot += plt.plot(self.rs[n:], self.extrapolated_values[n:], label="extrapolation of fine data")
        plt.title = title
        plt.legend()
        plt.show()

        n = len(self.fine_values)
        plot = plt.plot(self.rs[:n], self.fine_values, label="fine data")
        plot += plt.plot(self.rs, self.coarse_values, label="coarse data")
        plt.xlabel("Distance from midpoint [m]")
        plt.ylabel("Depth [m]")
        plot += plt.plot(self.rs[n:], self.extrapolated_values[n:], label="extrapolation of fine data")
        plot += plt.plot(self.rs, self.averaged_values, label="averaged depth")
        plt.title = title
        plt.legend()
        plt.show()
        return



if __name__ == "__main__":
    rs = list(range(20))
    fine_values = [ n**2 for n in range(10)]
    coarse_values = [ n for n in range(20) ]
    gamma=0.3

    ray=RadialSlice(rs, fine_values, coarse_values)
    ray.extrapolated_values = ray.polynomial_extrapolation(3)
    ray.averaged_values = ray.weighted_average(gamma)
    ray.plot()

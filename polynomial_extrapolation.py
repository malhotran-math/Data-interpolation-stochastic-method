import grid_classes
import read_data
import basic_functions
import numpy as np

def import_data(fname_fine, fname_coarse):
    xf, yf, vf = read_data.read_field(fname_fine)
    xc, yc, vc = read_data.read_field(fname_coarse)

    Rf = grid_classes.RectangularGrid(xf, yf, vf)
    Rc = grid_classes.RectangularGrid(xc, yc, vc)

    return Rf, Rc

def construct_radial_slices(Rf, Rc, n_rays, dr, r_min, r_max, inner_rel_dist, outer_rel_dist):
    inner_rel_steps = int(np.ceil(inner_rel_dist/dr))
    outer_rel_steps = int(np.ceil(outer_rel_dist/dr))


    Pf = grid_classes.polar_grid_from_rectangular(Rf, n_rays, dr, r_min, r_max)
    Pc = grid_classes.polar_grid_from_rectangular(Rc, n_rays, dr, r_min, r_max)
    # UG = grid_classes.polar_to_unsorted_grid(Pf)
    radial_slices = []
    for i in range(n_rays):
        rayf = Pf.v[i]
        rayc = Pc.v[i]
        cutoff = rayf.index(np.nan)
        #print(cutoff, inner_rel_steps, outer_rel_steps)
        rs = Pf.rs[cutoff-inner_rel_steps: cutoff + outer_rel_steps]
        fine_values = rayf[cutoff-inner_rel_steps: cutoff]
        coarse_values = rayc[cutoff-inner_rel_steps: cutoff + outer_rel_steps]
        rad_slice = grid_classes.RadialSlice(rs, fine_values, coarse_values)
        radial_slices.append((rad_slice, cutoff-inner_rel_steps))
    return radial_slices, Pf, Pc

def radial_slices_to_polar_grid(radial_slices, thetas, rs, center_x, center_y):
    n_rs = len(rs)
    n_thetas = len(thetas)
    v = []
    for rad_slice, start in radial_slices:
        v_rad = [np.nan for k in range(n_rs)]
        v_rad[start:start+len(rad_slice.averaged_values)] = rad_slice.averaged_values
        v.append(v_rad)
    Pa = grid_classes.PolarGrid(thetas, rs, v, center_x, center_y)
    return Pa


def merge_fine_coarse(Pf, Pc):
    n_thetas = len(Pf.thetas)
    v_new = []
    for i in range(n_thetas):
        v_new_theta = []
        end = Pf.v[i].index(np.nan)
        for j in range(len(Pc.rs)):
            if j < end:
                v_new_theta.append(Pf.v[i][j])
            else:
                v_new_theta.append(Pc.v[i][j])
        v_new.append(v_new_theta)
    return grid_classes.PolarGrid(Pf.thetas, Pc.rs, v_new, Pf.center_x, Pf.center_y)


def merge_fine_average_coarse(Pf, Pa, Pc):
    n_thetas = len(Pf.thetas)
    v_new = []
    for i in range(n_thetas):
        v_new_theta = []
        start = 0
        while np.isnan(Pa.v[i][start]):
            start += 1
        end = start
        while not np.isnan(Pa.v[i][end]):
            end += 1
        for j in range(len(Pc.rs)):
            if j < start:
                v_new_theta.append(Pf.v[i][j])
            elif j < end:
                v_new_theta.append(Pa.v[i][j])
            else:
                v_new_theta.append(Pc.v[i][j])
        v_new.append(v_new_theta)
    return grid_classes.PolarGrid(Pf.thetas, Pc.rs, v_new, Pf.center_x, Pf.center_y)


if __name__ == "__main__":
    deg = 2
    gamma = 1
    r_min = 0
    r_max = 65000
    dr = 100
    n_rays = 150
    inner_rel_distance = 2000
    outer_rel_distance = 4000

    Rf, Rc = import_data("high-res-bathymetry", "low-res-bathymetry")
    radial_slices, Pf, Pc = construct_radial_slices(Rf, Rc, n_rays, dr, r_min, r_max, inner_rel_distance, outer_rel_distance)
    Pf.plot()
    Pc.plot()
    Pm = merge_fine_coarse(Pf, Pc)
    Pm.plot()
    for rad_slice, cutoff in radial_slices:
        rad_slice.extrapolated_values = rad_slice.polynomial_extrapolation(deg)
        rad_slice.averaged_values = rad_slice.weighted_average(gamma)
    Pa = radial_slices_to_polar_grid(radial_slices, Pf.thetas, Pf.rs, Pf.center_x, Pf.center_y)
    Ptotal = merge_fine_average_coarse(Pf, Pa, Pc)
    Pa.plot()
    Ptotal.plot()
    for m in range(200):
        rad_slice, cutoff = radial_slices[m]
        print(m)
        rad_slice.plot(m)



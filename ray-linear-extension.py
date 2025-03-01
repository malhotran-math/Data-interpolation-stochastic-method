import numpy


def ray_linear(distances, height_ray, nb_points):
	n = len(distances)
	assert len(height_ray) == n
	relevant_distances = [ distances[r] for r in range(n-nb_points,n)]
	relevant_heights = [ height_ray[r] for r in range(n-nb_points,n)]
	a,b = numpy.polynomial.polynomial.Polynomial.fit(relevant_distances, relevant_heights, 1).convert().coef
	return a, b

def points_on_ray_linear(distances, height_ray,nb_relevant_points, distances_of_new_points):
	""" will create a data set containing the 'old' points on the ray as well as nb_new_points points generated
	with ray_linear """
	a,b=ray_linear(distances, height_ray, nb_relevant_points)
	distances_total=distances+distances_of_new_points
	new_heights=[ a+x*b for x in distances_of_new_points]
	heights_total=height_ray+new_heights
	return distances_total, heights_total

#print(ray_linear([1,2,3], [1, 1, 1], 2))

#print(points_on_ray_linear([1,2,3], [1, 1, 1], 2,[4,5,6]))

import bath.grid_classes as gc

# This function might not be used.
def complete_polar_grid_linearly(polar_grid, nb_relevant_points):
	#"""returns a polar grid whose values are extended with the chosen method """
	rs = polar_grid.rs
	for n in range(len(polar_grid.thetas)):
		heights = polar_grid.v[n]
		index_first_new = heights.index(None)
		old_heights = heights[:index_first_new]
		a, b = ray_linear(rs[:index_first_new], old_heights, nb_relevant_points)
		for m in range(index_first_new,len(rs)):
			polar_grid.v[n][m] = a+b*rs[m]
	return polar_grid




pol_grid = gc.PolarGrid([0,numpy.pi/2,numpy.pi,3*numpy.pi/2],[1,2,3,4],[[2,2,None,None],[4,5,4,None],[7,8,None,None],[10,11,12,None]], 0,0)

complete_polar_grid_linearly(pol_grid,2)

print(pol_grid.v)
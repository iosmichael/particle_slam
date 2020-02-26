import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
example map configuration

config = {
	# map configuration
	'res': 0.05,
	'size': 80,
	'free': -np.log(4),
	'occ': np.log(4),
	'occ_thres': 5,
	'logits_clip': 300,
}
"""

def coord2pix(x, config):
	'''
	the shape of x is: [..., 2]
	'''
	size = config['size']
	offset = np.array([size // 2, size // 2])
	return np.rint((x[..., :2] + offset) / config['res']).astype(np.int)

def map_update(x_body, pose, map2d, map_config):
	x_world = transform_body2world(x_body, pose)
	# remove the world points that has z above the ground level
	x_world = x_world[:2, x_world[2, :] > 0.05]
	x_map = coord2pix(x_world.T, map_config)
	temp_map = np.zeros(map2d.shape)

	# ray-tracing to update the free occupancy
	cv2.drawContours(temp_map, contours = [x_map], contourIdx = 0, color = map_config['free'], thickness = -1)
	# getting only the good index
	inds = np.logical_and.reduce([x_map[..., 0] >= 0, x_map[..., 0] < map2d.shape[0],
									x_map[..., 1] >= 0, x_map[..., 1] < map2d.shape[1]])
	x_map = x_map[inds, :]
	temp_map[x_map[:, 1], x_map[:, 0]] += map_config['occ'] - map_config['free']
	map2d += temp_map
	map2d = np.clip(map2d, -map_config['logits_clip'], map_config['logits_clip'])
	return map2d

def get_corr_optimized(x_body, particles, map2d, map_config):
	'''
	the shape of the body: 4 * num_occ
	the shape of the particles: 3 * num_particle
	the shape of the map: num_x * num_y
	'''
	n = particles.shape[1]
	transform_body2world_optimized = np.zeros((n, 4, 4))

	cyaw, syaw = np.cos(particles[-1, :]), np.sin(particles[-1, :])

	# construct the optimized transformation matrix
	transform_body2world_optimized[:, 0, 0] = cyaw
	transform_body2world_optimized[:, 1, 1] = cyaw
	transform_body2world_optimized[:, 0, 1] = -syaw
	transform_body2world_optimized[:, 1, 0] = syaw
	transform_body2world_optimized[:, 0, 3] = particles[0, :]
	transform_body2world_optimized[:, 1, 3] = particles[1, :]
	transform_body2world_optimized[:, -1, -1] = 1

	# get world transformations
	x_world = np.expand_dims(transform_body2world_optimized, axis=1) @ np.expand_dims(np.expand_dims(x_body.T, axis=-1), axis=0)
	
	# x_world shape: num_particles * num_ray * 4
	x_map = coord2pix(x_world.squeeze(-1)[...,:2], map_config)
	return np.sum(map2d[x_map[...,1],x_map[...,0]] > map_config['occ_thres'], axis=1)

'''
different transformation functions:
1. head frame
2. body frame
3. world frame
'''

def transform_lidar(ranges, clip_threshold=30):
	ranges = np.clip(ranges, 0.1, clip_threshold)
	angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
	x = ranges.T * np.cos(angles) 
	y = ranges.T * np.sin(angles)
	# lidar to head is 15 cm
	z = np.zeros(x.shape)
	X = np.hstack([x, y, z, np.ones(x.shape)]).T
	X = X[:, np.linalg.norm(X[:2, :], axis=0) >= 0.1]
	T_lidar2head = np.array([[1, 0, 0, 0],
							 [0, 1, 0, 0],
							 [0, 0, 1, 0.15],
							 [0, 0, 0, 1]])
	return T_lidar2head @ X

def transform_head2body(x, yaw, pitch):
	# x shape: (4, n) homogenous coordinates
	assert x.shape[0] == 4
	R = yawpitch2R(yaw, pitch)
	t = np.array([[0], [0], [0.33]])
	T_head2body = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])
	return T_head2body @ x

def transform_body2world(x, pose):
	# x shape: (4, n) homogenous coordinates
	assert x.shape[0] == 4
	# 3D to 2D transform
	px, py, theta = pose[0, 0], pose[1, 0], pose[2, 0]
	# yaw angle for the particle
	R = np.array([[np.cos(theta), -np.sin(theta), 0], 
					[np.sin(theta), np.cos(theta), 0], 
					[0, 0, 1]])
	t = np.array([[px], [py], [0.93]])
	T_body2world = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])
	return T_body2world @ x

def transform_lidar2body(yaw, pitch, pose):
	T_lidar2head = np.array([[1, 0, 0, 0],
							 [0, 1, 0, 0],
							 [0, 0, 1, 0.15],
							 [0, 0, 0, 1]])
	R = yawpitch2R(yaw, pitch)
	t = np.array([[0], [0], [0.33]])
	T_head2body = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])
	x, y, theta = pose[0, 0], pose[1, 0], pose[2, 0]
	R_w = np.array([[np.cos(theta), -np.sin(theta), 0], 
					[np.sin(theta), np.cos(theta), 0], 
					[0, 0, 1]])
	t_w = np.array([[x], [y], [0.93]])
	T_body2world = np.vstack([np.hstack([R_w, t_w]), np.array([0, 0, 0, 1])])
	return T_body2world @ T_head2body

def yawpitch2R(yaw, pitch):
	R_y = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
					[np.sin(yaw), np.cos(yaw), 0], 
					[0, 0, 1]])
	R_p = np.array([[np.cos(pitch), 0, np.sin(pitch)],
					[0, 1, 0], 
					[-np.sin(pitch), 0, np.cos(pitch)]])
	R = R_y @ R_p
	return R

'''
really slow stuff that doesn't give good quality particles

def get_correlation(target, p_world, x_im, y_im):
	# shifting around a 5 x 5 search grid
	# xy_range = [-0.1 , -0.05,  0.  ,  0.05,  0.1]
	x_range = np.arange(-0.1,0.1+0.05,0.05)
	y_range = np.arange(-0.1,0.1+0.05,0.05)
	# robust_angles = np.array([-0.002, 0, 0.002])
	robust_angles = np.array([-0.005, 0, 0.005])
	corr = mapCorrelation(target, x_im, y_im, p_world, x_range, y_range, robust_angles)
	ind = np.argmax(corr)
	a_ind = ind // 25
	x_ind = (ind % 25) // 5
	y_ind = (ind % 25) % 5
	# print(f'offset: {np.array([robust_angles[a_ind], x_range[x_ind],y_range[y_ind]])}')
	assert np.max(corr) == corr[a_ind, x_ind, y_ind]
	return np.max(corr), np.array([x_range[x_ind], y_range[y_ind], robust_angles[a_ind]])

def mapCorrelation(im, x_im, y_im, vp, xs, ys, thetas):
	nx = im.shape[0]
	ny = im.shape[1]
	na = thetas.shape[0]
	xmin = x_im[0]
	xmax = x_im[-1]
	xresolution = (xmax-xmin)/(nx-1)
	ymin = y_im[0]
	ymax = y_im[-1]
	yresolution = (ymax-ymin)/(ny-1)
	nxs = xs.size
	nys = ys.size
	cpr = np.zeros((na, nxs, nys))
	for i, yaw in enumerate(thetas):
		R = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
		vrp = R @ vp[:2, :]
		for jy in range(0,nys):
			y1 = vrp[1,:] + ys[jy] # 1 x 1076
			iy = np.int16(np.round((y1-ymin)/yresolution))
			for jx in range(0,nxs):
				x1 = vrp[0,:] + xs[jx] # 1 x 1076
				ix = np.int16(np.round((x1-xmin)/xresolution))
				valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), 
											np.logical_and((ix >=0), (ix < nx)))
				cpr[i,jx,jy] = np.sum(im[ix[valid],iy[valid]])
	return cpr
'''
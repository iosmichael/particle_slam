import numpy as np
from THOR.load_data import *

def transform_ir2kinect(X):
	"""
	x has the shape of 4 * N, where N is the number of pixels
	"""
	T_ir2rbg = np.hstack([getExtrinsics_IR_RGB()['rgb_R_ir'], getExtrinsics_IR_RGB()['rgb_T_ir'].reshape(-1,1)])
	T_ir2rbg = np.vstack([T_ir2rbg, np.array([0,0,0,1])])
	return T_ir2rbg @ X

def transform_kinect2head(X):
	T_kinect2head = np.array([[1, 0, 0, 0],
							 [0, 1, 0, 0],
							 [0, 0, 1, 0.07],
							 [0, 0, 0, 1]])
	return T_kinect2head @ X

def get_K(is_rgb=True):
	data = getIRCalib()
	if is_rgb:
		data = getRGBCalib()
	f, c = np.diag(data['fc']), data['cc'].reshape(-1,1)
	K = np.vstack([np.hstack([f,c]), np.array([0, 0, 1])])
	return K

def get_uv1(img):
	pts = np.ones((img.shape[0], img.shape[1], 3))
	# img shape 1 is x (u), depth shape 0 is y (v)
	pts[..., 1] = np.tile(np.arange(img.shape[0]).reshape(-1,1), (1, img.shape[1]))
	pts[..., 0] = np.tile(np.arange(img.shape[1]).reshape(1,-1), (img.shape[0], 1))
	assert pts[0, 10, 0] == 10
	return pts

def get_depth2cam(depth, image, K_ir, K_rgb, T_ir2rgb):
	"""
	:param - the depth (r, c) from the kinect image camera, and the calibration matrix of the image camera, the principal point of the image
	:return - 3D point clouds with X: (N * 4)
	"""
	K_irv = np.linalg.inv(K_ir)
	print(f"{K_ir} \n{K_irv}")
	uv1 = get_uv1(depth)
	X = K_irv[np.newaxis, np.newaxis, ...] @ uv1[..., :3, np.newaxis]
	# K^-1 [x, y, 1] * Z = [X, Y, Z]
	X[..., :3] = X[..., :3] * depth[..., np.newaxis] / 1000
	assert np.allclose(X[..., 2], depth * 0.001)
	# homogenize the 3d points
	X[..., np.newaxis] = 1
	# transformation from depth optical frame to camera optical frame
	X = T_ir2rbg @ X.T
	# [X', Y', Z']
	uv_rgb = K_rgb @ X[:3, :]
	uv_rgb2 = (uv_rgb / uv_rgb[-1, :])[:2, :].astype(np.int)
	# 2 x n

	# uv1_rgb: h x w x 3
	uv1_rgb = get_uv1(image)
	# xyz rgb
	uv1_rgb = uv1_rgb[uv_rgb2[1, :], uv_rgb2[0, :], :].reshape(-1, 6).T

	return X.reshape(-1, 4).T

def perspective_projection(x, K, R_c, p_c):
	"""
	from lecture slide 7 in ECE276A observation model
	:param - x, [x,y,z] coordinates in the world frame 3 * N
		   - K, camera calibration matrix, R_c|p_c, rotation and position of the camera in the world frame
	:return - u, v, 1 pixel coordinates in the camera frame
	"""
	# rotation from regular to optical frame
	R_o = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
	P = np.array([[1, 0, 0], [0, 1, 0]])
	return P @ K @ R_o @ R_c.T @ (x - p_c.reshape(3,1))

def from_optical2world(x_o):
	"""
	inverse the optical rotation matrix
	:param - x_o, [X, Y, Z] in the optical frame
	"""
	T_o = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
	return np.linalg.inv(T_o) @ x_o

def transform_cam2world(yaw, pitch, pose):
	"""
	get the rotation and translation matrix for the camera in the world frame
	"""
	T_kinect2head = np.array([[1, 0, 0, 0],
							 [0, 1, 0, 0],
							 [0, 0, 1, 0.07],
							 [0, 0, 0, 1]])
	T_head_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], 
					[np.sin(yaw), np.cos(yaw), 0, 0], 
					[0, 0, 1, 0],
					[0, 0, 0, 1]])
	T_head_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
					[0, 1, 0, 0], 
					[-np.sin(pitch), 0, np.cos(pitch),0],
					[0, 0, 0, 1]])

	T_head2body = T_head_yaw @ T_head_pitch
	# 3D to 2D transform
	px, py, theta = pose[0, 0], pose[1, 0], pose[2, 0]
	# yaw angle for the particle

	T_body2world = np.array([[np.cos(theta), -np.sin(theta), 0,   px],
							 [np.sin(theta), np.cos(theta),  0,   py],
							 [			  0, 			 0,  1, 0.93],
							 [			  0, 			 0,  0,    1]])

	T_cam2world = T_body2world @ T_head2body @ T_kinect2head
	assert np.allclose(np.linalg.det(T_cam2world[:3, :3]), 1)
	return T_cam2world[:3, :3], T_cam2world[-1, :3]

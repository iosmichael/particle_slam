import numpy as np
from THOR.load_data import *

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

def get_depth2cam(depth, image, K_ir, K_rgb):
	"""
	:param - the depth (r, c) from the kinect image camera, and the calibration matrix of the image camera, the principal point of the image
	:return - 3D point clouds with X: (N * 4)
	"""
	uv_d = get_uv1(depth) # uv_d (IMG_HEIGHT * IMG_WIDTH * 4)
	T = np.zeros((4,4))
	T[:3, :3], T[:3, -1], T[-1, -1] = getExtrinsics_IR_RGB()['rgb_R_ir'], getExtrinsics_IR_RGB()['rgb_T_ir'], 1
	xy_i = np.linalg.inv(K_ir) @ uv_d.reshape(-1, 3).T
	
	# get pixel colors through homography transformation
	pix_pts = K_rgb @ (T[:3, :3] @ xy_i + T[:3,-1].reshape(-1,1))
	pix_pts /= pix_pts[-1, :] # dehomogenize
	pix_pts = pix_pts.astype(np.int)

	# filter out bad points outside of image
	inds = np.logical_and(np.logical_and(pix_pts[1, :] >= 0, pix_pts[1, :] < image.shape[0]),
						  np.logical_and(pix_pts[0, :] >= 0, pix_pts[0, :] < image.shape[1]))
	pix_color = image[pix_pts[1, inds], pix_pts[0, inds], :].reshape(-1, 3).T # 3 * N

	xy_i *= (depth.reshape(1, -1) / 1000)

	xy_i = xy_i[:, inds]
	xy_i = np.vstack((xy_i, np.ones((1, xy_i.shape[1])))) # homogenize xy
	X_cam_pts = T @ xy_i # 4 * N
	assert X_cam_pts.shape[1] == pix_color.shape[1]
	return X_cam_pts, pix_color

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

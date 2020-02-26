import numpy as np 
from matplotlib import pyplot as plt
from THOR.load_data import *

import scipy.ndimage

def visualize_cumulative_noise():
	poses = []
	np.random.seed(0)
	pose = np.zeros((3,1))
	for i in range(10000):
		poses.append(pose)
		pose = pose + (np.random.multivariate_normal(np.zeros(3), np.diag(np.ones(3)), size=1) * 1e-3).T
		assert pose.shape == (3, 1)
	poses = np.concatenate(poses, axis=1)
	plt.plot(poses[-1, :])
	plt.show()

def visualize_preprocessing_imu(k=10):
	lidar = get_lidar("THOR/lidar/train_lidar4")
	poses = []
	for i in range(len(lidar)):
		poses.append(lidar[i]['delta_pose'].reshape(3,1))
	poses = np.concatenate(poses, axis=1)
	print(poses.shape)
	plt.plot(poses[0, :])
	plt.show()
	poses = scipy.ndimage.filters.convolve1d(poses, np.ones(k)/k, axis=1)
	print(poses.shape)
	plt.show()

def visualize_synchronization():
	lidar = get_lidar("THOR/lidar/train_lidar4")
	joint = get_joint("THOR/joint/train_joint4")
	print("===robot statistics===")
	print(f"joint: {len(joint['ts'])}")
	print(f"lidar: {len(lidar)}")

	print(np.max(joint['ts']))
	timestamps = []
	for i in range(len(lidar)):
		timestamps.append(lidar[i]['t'])
	print(np.max(np.array(timestamps)))
	
	print('lidar behind joint ? ', np.max(np.array(timestamps)) > np.max(joint['ts']))
	print('lidar before joint ? ', np.min(np.array(timestamps)) < np.min(joint['ts']))

if __name__ == '__main__':
	visualize_preprocessing_imu()

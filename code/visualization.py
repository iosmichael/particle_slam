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

def visualize_dataset():
	index = [0, 1, 2, 3, 4]
	for i in index:
		lidar = get_lidar(f"THOR/lidar/train_lidar{i}")
		joint = get_joint(f"THOR/joint/train_joint{i}")
		print(f"===robot statistics on dataset {i}===")
		print(f"\tjoint: {len(joint['ts'][0])}")
		print(f"\tlidar: {len(lidar)}")

if __name__ == '__main__':
	visualize_dataset()

import numpy as np
from matplotlib import pyplot as plt
from texture_utils import *
from utils import *

plt.figure(figsize=(6,6))
plt.tight_layout()

config = {
	# experiment dataset
	'cam': 'THOR/cam/RGB_0',
	'ir': 'THOR/cam/DEPTH_0',
	'joint': 'THOR/joint/train_joint4',

	# experiment configuration
	'dirname': 'lidar4_5000',
	'expr_name': 'lidar4_particle100',
	'plot_title': 'lidar4 particle 5000 map',

	# map configuration
	'res': 0.05,
	'size': 80,
	'free': -np.log(4),
	'occ': np.log(4),
	'occ_thres': 5,
	'logits_clip': 300,
}

np.random.seed(10)

def main():
	K_ir, K_rgb = get_K(is_rgb=False), get_K(is_rgb=True)

	img_data = dict(np.load("data/lidar0_particle100_data.npy", allow_pickle=True).item())
	pose3d, pose2d, map2d, ts = img_data['pose3d'], img_data['poses2d'], img_data['map'], img_data['ts']

	r0 = get_rgb(config['cam'])
	d0 = get_depth(config['ir'])
	joint = get_joint(config['joint'])
	joint_ts = joint['ts']

	for img, depth in zip(r0, d0):

		joint_ind = np.argmin(np.absolute(joint_ts - img['t']))
		pose3d_ind = np.argmin(np.absolute(ts - img['t']))
		yaw, pitch = joint['head_angles'][0, joint_ind], joint['head_angles'][1, joint_ind]
		pose = pose3d[:, pose3d_ind].reshape(-1, 1)
		assert pose.shape == (3, 1)

		# matching the image with a specific pose from the robot estimation
		# X_d has shape of 4 * N
		depth_pix = get_pix2IR(depth['depth'], K_ir)
		
		# plt.imshow(depth['depth'], cmap='gray')
		# plt.show()

		# X_d in kinect camera frame
		x = transform_ir2kinect(depth_pix)
		x = from_optical2world(x)
		print(f"kinect min z: {np.min(x[-2, :])}")
		# transform X_d to X_i and apply the pose transformation
		x = transform_kinect2head(x)
		print(f"head min z: {np.min(x[-2, :])}")
		x = transform_head2body(x, yaw, pitch)
		print(f"body min z: {np.min(x[-2, :])}")
		x = transform_body2world(x, pose)
		print(f"world min z: {np.min(x[-2, :])}")
		x_map = coord2pix(x.T, config)
		print(f"max: {np.max(x_map[..., 0])} min: {np.min(x_map[...,1])}")

		map2d_tex = np.zeros(map2d.shape)
		print(x_map.shape)
		inds = np.logical_and(np.logical_and(x_map[..., 1] >= 0, x_map[..., 1] < map2d.shape[0]),
							   np.logical_and(x_map[..., 0] >= 0, x_map[..., 0] < map2d.shape[1]))
		print(np.sum(inds))
		assert False
		map2d_tex[x_map[inds, 0], x_map[inds, 1]] = 1
		plt.imshow(map2d_tex, cmap='gray')
		plt.show()
		# assert False
		color_image = img['image']
		# plt.imshow(inds, cmap='gray')
		# plt.show()
		world_map = np.zeros(color_image.shape)
		img_pix = dd[:3, X_w[2, :] < -0.2]
		print(img_pix)
		K_rgb = get_K(is_rgb=True)
		uv = (K_rgb @ img_pix) / (K_rgb @ img_pix)[-1, :]
		# get transformation
		uv = uv.astype(np.int)
		inds = np.logical_and.reduce([uv[1, ...] >= 0, uv[1, ...] < color_image.shape[0],
									  uv[0, ...] >= 0, uv[0, ...] < color_image.shape[1]])

		world_map[uv[1, inds], uv[0, inds]] = color_image[uv[1, inds], uv[0, inds]]

		plt.imshow(world_map/255)
		plt.show()
		# plt.imshow(img['image'])
		# plt.show()

if __name__ == '__main__':
	main()

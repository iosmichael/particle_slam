import numpy as np
from matplotlib import pyplot as plt
from texture_utils import *
from utils import *
from tqdm import tqdm

plt.figure(figsize=(6,6))
plt.tight_layout()

config = {
	# experiment dataset
	'cam': 'THOR/cam/RGB_3',
	'ir': 'THOR/cam/DEPTH_3',
	'joint': 'THOR/joint/train_joint3',

	# experiment configuration
	'dirname': 'texture',
	'saved_data_path': 'data/lidar3_particle2000_data.npy',
	'expr_name': 'lidar3_texture',
	'plot_title': 'lidar3 particle slam texture map',

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

	img_data = dict(np.load(config['saved_data_path'], allow_pickle=True).item())
	pose3d, pose2d, map2d, ts = img_data['pose3d'], img_data['poses2d'], img_data['map'], img_data['ts']

	r0 = get_rgb(config['cam'])
	d0 = get_depth(config['ir'])
	joint = get_joint(config['joint'])
	joint_ts = joint['ts']

	map2d_tex = np.zeros((map2d.shape[0], map2d.shape[1], 3))

	for img, depth in tqdm(zip(r0, d0)):

		joint_ind = np.argmin(np.absolute(joint_ts - img['t']))
		pose3d_ind = np.argmin(np.absolute(ts - img['t']))
		yaw, pitch = joint['head_angles'][0, joint_ind], joint['head_angles'][1, joint_ind]
		pose = pose3d[:, pose3d_ind].reshape(-1, 1)
		assert pose.shape == (3, 1)

		# matching the image with a specific pose from the robot estimation
		# X_d has shape of 4 * N
		x, color = get_depth2cam(depth['depth'], img['image'], K_ir, K_rgb)
		x = from_optical2world(x)
		# transform X_d to X_i and apply the pose transformation
		x = transform_kinect2head(x)
		x = transform_head2body(x, yaw, pitch)
		x = transform_body2world(x, pose)
		x_map = coord2pix(x.T, config) # 2 * N

		inds = np.logical_and(np.logical_and(x_map[..., 1] >= 0, x_map[..., 1] < map2d.shape[0]),
							  np.logical_and(x_map[..., 0] >= 0, x_map[..., 0] < map2d.shape[1]))
		color = color[:, inds]
		x_map = x_map[inds, :].T
		x = x[:, inds]

		color = color[:, x[2, :] < -0.1]
		x_map = x_map[:, x[2, :] < -0.1]
		assert x_map.shape[1] == color.shape[1]

		map2d_tex[x_map[1, :], x_map[0, :], :] = color.T

	# clipping based on logits map
	binary = (map2d < 0).astype(np.int)
	# save the data into texture directory
	plt.imshow(map2d_tex * binary[..., np.newaxis] / 255)
	plt.scatter(x=pose2d[:, 0], y=pose2d[:, 1], s=5, c='g', marker='.')
	plt.title(f"{config['plot_title']}")
	plt.savefig(f"{config['dirname']}/{config['expr_name']}_texture.png")
	print("===Successfully Saved Texture===")

if __name__ == '__main__':
	main()

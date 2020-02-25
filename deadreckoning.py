from matplotlib import pyplot as plt
from tqdm import tqdm
import cProfile
from THOR.load_data import *
from utils import *
import yaml

import scipy.ndimage
from scipy.special import logsumexp

plt.figure(figsize=(6,6))
plt.tight_layout()

config = {
	# experiment dataset
	'lidar': 'THOR/lidar/train_lidar0',
	'joint': 'THOR/joint/train_joint0',

	# experiment configuration
	'dirname': 'records',
	'expr_name': 'lidar0_particle100',
	'plot_title': 'lidar0 particle 100 map',

	# map configuration
	'res': 0.1,
	'size': 40,
	'free': -np.log(4),
	'occ': np.log(4),
	'occ_thres': 0,
	'logits_clip': 200,

	# number of particles
	'num_particles': 1,
	'resample_thres': .5
}

np.random.seed(10)

def main():
	# config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.SafeLoader)
	system = SLAMSystem(config)
	# read sensor data
	lidar = get_lidar(config['lidar'])
	joint = get_joint(config['joint'])
	joint_ts = joint['ts']
	'''
	calculate the statistics of the dataset
	'''
	print(f"dataset statistics: \n\tlidar frames: {len(lidar)} \n\tjoint frames: {joint_ts.shape[1]}")

	# main loop
	for f in tqdm(range(1500,len(lidar))):
		'''
		data synchronization process: matching joint timestamp with the lidar timestamp
		'''
		joint_ind = np.argmin(np.absolute(joint_ts - lidar[f]['t'][0][0]))
		system.update_loop(lidar[f], joint['head_angles'][:, joint_ind])
		if f % 1000 == 0:
			system.map_show(f)
	system.map_show(f)

class SLAMSystem(object):
	def __init__(self, config):
		self.config = config

		size, res = config['size'], config['res']
		self.map2d = np.zeros((int(size // res), int(size // res))).astype(np.float32)

		# initialize the particles
		N = config['num_particles']
		self.particles = np.zeros((3, N))
		self.W = np.ones(N) / N

		# resampling of the weight criterion
		self.resample_thres = N * config['resample_thres']

		self.episodes = 0
		# system statistics
		self.correlations = []
		self.effs = []

	def motion_model(self, delta_pose):
		# sensor noise metric
		self.particles[0, :] = self.particles[0, :] + delta_pose[0, 0]
		self.particles[1, :] = self.particles[1, :] + delta_pose[1, 0]
		self.particles[2, :] = self.particles[2, :] + delta_pose[2, 0]
		# noise = (np.random.multivariate_normal(np.zeros(3), np.diag(np.ones(3)).astype(np.float32), tol=1e-5, size=self.particles.shape[1])).T
		# noise = np.random.randn(3, self.particles.shape[1]) * 5e-3

	def update_loop(self, lidar, joint):
		ranges = lidar['scan']
		delta_pose = lidar['delta_pose'].reshape(3,1)
		yaw, pitch = joint[0], joint[1]
		x = transform_lidar(ranges, clip_threshold=15)
		x = transform_head2body(x, yaw, pitch)
		if np.max(self.map2d) <= self.config['occ_thres']:
			self.map2d = map_update(x, self.particles[:, np.argmax(self.W)].reshape(-1,1), self.map2d, self.config)
		else:
			self.motion_model(delta_pose)
			self.map2d = map_update(x, self.particles[:, np.argmax(self.W)].reshape(-1,1), self.map2d, self.config)
		self.episodes += 1

	def map_show(self, ep):
		plt.imshow(self.map2d, cmap="RdBu", interpolation='none')
		# plt.scatter(x=poses[0, :], y=poses[1, :], s=.05, c='g', marker='.')
		# plt.scatter(x=self.delta_poses[1, :], y=self.delta_poses[0, :], s=.05, c='r', marker='x')
		plt.title(f"{self.config['plot_title']}")
		plt.savefig(f"{self.config['dirname']}/{self.config['expr_name']}_map_{ep}.png")


if __name__ == '__main__':
	# cProfile.run('main()','restats.txt')
	main()
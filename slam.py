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
	'lidar': 'THOR/lidar/train_lidar4',
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

	# number of particles
	'num_particles': 5000,
	'resample_thres': .4
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

	for f in tqdm(range(0,len(lidar))):
		'''
		data synchronization process: matching joint timestamp with the lidar timestamp
		'''
		joint_ind = np.argmin(np.absolute(joint_ts - lidar[f]['t'][0][0]))
		system.update_loop(lidar[f], joint['head_angles'][:, joint_ind])
		if f % 1000 == 0 and f != 0:
			system.map_show(f)
	# system.map_show(f)
	system.save_data()

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
		self.poses = []
		self.ts = []
		self.episodes = 0
		# system statistics
		self.correlations = []
		self.effs = []

	def motion_model(self, delta_pose):
		# sensor noise metric
		self.particles[0, :] = self.particles[0, :] + delta_pose[0]
		self.particles[1, :] = self.particles[1, :] + delta_pose[1]
		self.particles[2, :] = self.particles[2, :] + delta_pose[2]
		# noise = (np.random.multivariate_normal(np.zeros(3), np.diag(np.ones(3)).astype(np.float32), tol=1e-5, size=self.particles.shape[1])).T
		noise = np.random.randn(3, self.particles.shape[1]).astype(np.float) * 0.01
		assert noise.shape == self.particles.shape
		self.particles = self.particles + noise

	def update_loop(self, lidar, joint):
		ranges = lidar['scan']
		delta_pose = lidar['delta_pose'].reshape(3,1)
		yaw, pitch = joint[0], joint[1]
		x = transform_lidar(ranges, clip_threshold=20)
		x = transform_head2body(x, yaw, pitch)
		if np.max(self.map2d) <= self.config['occ_thres']:
			self.map2d = map_update(x, self.particles[:, np.argmax(self.W)].reshape(-1,1), self.map2d, self.config)
		else:
			self.motion_model(delta_pose)
			if self.episodes % 5 != 0:
				# apply the motion model
				self.episodes += 1
				return
			# apply the map correlation, this implementation allows huge batch of particles
			scores = get_corr_optimized(x, self.particles, self.map2d, self.config)
			
			# reweight the particles weight
			self.reweight_particles(scores)
			self.poses.append(self.particles[:, np.argmax(self.W)].reshape(-1,1))
			self.ts.append(lidar['t'][0, 0])
			# perform map update
			eff = np.sum(self.W) ** 2 / np.sum(self.W ** 2)
			if eff <= self.config['resample_thres'] * self.config['num_particles']:
				self.map2d = map_update(x, self.particles[:, np.argmax(self.W)].reshape(-1,1), self.map2d, self.config)

			# resample based on the criterion with the particle weights
			self.stratified_resample()
		self.episodes += 1

	def reweight_particles(self, scores):
		scores = scores - np.max(scores)
		self.W = np.exp(scores) * self.W
		self.W /= np.sum(self.W)

	# stratified resampling of the particles
	def stratified_resample(self):
		eff = np.sum(self.W)**2 / np.sum(self.W ** 2)
		self.effs.append(eff)
		if eff <= self.resample_thres:
			N = self.config['num_particles']
			temp_particles = np.zeros((3, N))
			c_weights, n = np.cumsum(self.W), 0
			for i in range(N):
				sample_noise = np.random.uniform(0, 1 / N)
				sample_choice = sample_noise + i * (1 / N)
				while sample_choice > c_weights[n]:
					n = n + 1
				temp_particles[:, i] = self.particles[:, n]
			self.particles = temp_particles
			self.W = np.ones(N) / N

	def map2binary(self):
		# given a 2D map of logits return a binary map with -1 being free and 1 being occupied
		return (m > self.config['occ_thres']).astype(np.int)

	def map_show(self, ep):
		plt.clf()
		poses = coord2pix(np.concatenate(self.poses, axis=1)[:2, :].T, self.config)
		plt.imshow(self.map2d, cmap="gray", interpolation='none')
		plt.scatter(x=poses[:, 0], y=poses[:, 1], s=5, c='g', marker='.')
		plt.title(f"{self.config['plot_title']}")
		plt.savefig(f"{self.config['dirname']}/{self.config['expr_name']}_map_{ep}.png")

	def save_data(self):
		poses3d = np.concatenate(self.poses, axis=1)
		ts = np.array(self.ts)
		poses2d = coord2pix(np.concatenate(self.poses, axis=1)[:2, :].T, self.config)
		data = {
			'pose3d': poses3d,
			'poses2d': poses2d,
			'map': self.map2d,
			'ts': ts
		}
		np.save(f"{self.config['dirname']}/{self.config['expr_name']}_data.npy", data)
		print(f"successfully saved data for {self.config['expr_name']}")
		print(f"data statistics: poses3d: {poses3d.shape}, poses2d: {poses2d.shape}, map: {self.map2d.shape}, ts: {ts.shape}")

	# def stats_draw(self, ep):
	# 	plt.clf()
	# 	plt.plot(self.correlations, c='g')
	# 	plt.savefig(f'records/{expr_name}_corr_{ep}.png')

if __name__ == '__main__':
	# cProfile.run('main()','restats.txt')
	main()
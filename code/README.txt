README

====== FILE DESCRIPTION ============

slam.py - the python execution file for running Particle SLAM, which takes dataset from dirname in the configuration dictionary

utils.py - contains the transformation and functions used in particle slam estimation

deadreckoning.py - the python execution file specifically used for running dead-reckoning without particle filters

texture.py - the python execution file takes the saved result from the slam.py and overlay textures onto the result

texture_utils.py - contains transformation and functions specific to the camera projection and texture overlay

====== DATASET DESCRIPTION ==========

The dataset used the data collected by the "THOR" humanoid robot
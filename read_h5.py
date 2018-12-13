import h5py

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

filename = '/media/amr/Amr_4TB/Sergio_All_Videos/Results_from_Cluster/R278_OR_1h_delay_Exp_2017_11_19_R250_R278_newone_20171119_145517.avi_compressed.h5'

f = h5py.File(filename, 'r')

print("Keys: %s" % f.keys())

a_group_key = list(f.keys())[0]

data = pd.read_hdf(filename, a_group_key)

x = data.iloc[330:, 0][data.iloc[:,2]>0.7] # Remove the first few frames to avoid tracking the hand of the experimenter 
y = data.iloc[330:, 1][data.iloc[:,2]>0.7]


#create trajectory
#you can control the outliers by modifying the likelihood value [data.iloc[:,2]>0.7]
plt.figure(figsize=(4,4), dpi=300)
plt.plot(x, y, color='k', linewidth=1)
plt.axis('off')
plt.gca().invert_yaxis() #otherwise the images appear mirror imaged
plt.savefig('/media/amr/Amr_4TB/Sergio_All_Videos/Results_from_Cluster/trajectory.png')
# plt.show()


#Create density maps
from scipy.stats import kde


# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
nbins=300
k = kde.gaussian_kde([x,y])
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))


# Make the plot
plt.figure(figsize=(6,4), dpi=300)
plt.pcolormesh(xi, yi, (zi.reshape(xi.shape) - zi.min())/(zi.max()), cmap='jet') #normalize zi value to get colorbar from 0-1
plt.colorbar(ticks=[0,0.2,0.4,0.6,0.8,1])
plt.axis('off')
plt.gca().invert_yaxis() #otherwise the images appear mirror imaged
plt.savefig('/media/amr/Amr_4TB/Sergio_All_Videos/Results_from_Cluster/densitymap.png')
# plt.show()


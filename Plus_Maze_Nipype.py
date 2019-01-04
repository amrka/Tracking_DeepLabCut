#analyzing plus maze behavior video
from nipype import config
cfg = dict(execution={'remove_unnecessary_outputs': False})
config.update_config(cfg)

import numpy as np
import matplotlib.pyplot as plt
import nipype.interfaces.utility as utility
from nipype.interfaces.utility import IdentityInterface, Function
from os.path import join as opj
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode

import deeplabcut
import cv2
import os

#------------------------------------------------------------------------------------------------------------------
# In[2]:

experiment_dir = '/home/in/aeed/Work/October_Acquistion/' 

subject_list = ['229', '230', '232', '233', 
                '234', '235', '237', '242', 
                '243', '244', '245', '252', 
                '253', '255', '261', '262', 
                '263', '264', '273', '274', 
                '281', '282', '286', '287', 
                '362', '363', '364', '365', 
                '366']

# subject_list = ['229', '230', '365', '274']
                
# subject_list = ['230', '365']


output_dir  = 'Plus_Maze_output'
working_dir = 'Plus_Maze_workingdir'

Plus_Maze_workflow = Workflow (name = 'Plus_Maze_workflow')
Plus_Maze_workflow.base_dir = opj(experiment_dir, working_dir)

#-----------------------------------------------------------------------------------------------------
# In[3]:


# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

#-----------------------------------------------------------------------------------------------------
# In[4]:

templates = {

             'plus_maze'  : 'Data/{subject_id}/plus_maze_{subject_id}.avi'
 }

selectfiles = Node(SelectFiles(templates,
                               base_directory=experiment_dir),
                   name="selectfiles")

#-----------------------------------------------------------------------------------------------------
# In[5]:

# datasink = Node(DataSink(base_directory=experiment_dir,
#                          container=output_dir),
#                 name="datasink")
datasink = Node(DataSink(), name = 'datasink')
datasink.inputs.container = output_dir
datasink.inputs.base_directory = experiment_dir

substitutions = [('_subject_id_', '')]

datasink.inputs.substitutions = substitutions

#-----------------------------------------------------------------------------------------------------
# In[6]:


def deeplabcut(video):
        import deeplabcut
        import os
        import glob
        from shutil import copy2

        path_config = '/home/in/aeed/plus_maize/Plus_Maiz-Amr-2018-12-25/config.yaml'

        deeplabcut.analyze_videos(path_config, [video], save_as_csv=True)

        deeplabcut.create_labeled_video(path_config, [video])


        h5_file = os.path.abspath(glob.glob('*.h5')[0])
        labeled_video = os.path.abspath(glob.glob('*_labeled.mp4')[0])

        pickle = os.path.abspath(glob.glob('*.pickle')[0])
        csv_file = os.path.abspath(glob.glob('*.csv')[0])

        #I failed to force deeplabcut to put the output in the workflow directory, so I am copying them
        #like a caveman

        subj_no = os.getcwd()[-3:]
        experiment_dir = '/home/in/aeed/Work/October_Acquistion/'
        working_dir = 'Plus_Maze_workingdir'

        copy2(h5_file, '%s%s/Plus_Maze_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))
        copy2(labeled_video,'%s%s/Plus_Maze_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))
        copy2(pickle,'%s%s/Plus_Maze_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))
        copy2(csv_file,'%s%s/Plus_Maze_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))

        return  h5_file, labeled_video


          


deeplabcut = Node(name = 'DeepLabCut',
                  interface = Function(input_names = ['video'],
                  output_names = ['h5_file', 'labeled_video'],
                  function = deeplabcut))






#-----------------------------------------------------------------------------------------------------
# In[7]:
#Draw Trajectory

def draw_trajectory(h5_file, video):
		import os
		import h5py #top open h5 binary files
		import matplotlib.pyplot as plt
		import matplotlib
		import pandas as pd
		import numpy as np
		import cv2 
		import glob
		np.set_printoptions(suppress=True) 

		#Get fps to use it for indexing, we need only 10 min from beg
		#so the indexing will be from 0:
		cap = cv2.VideoCapture(video)

		fps = int(cap.get(cv2.CAP_PROP_FPS))

		ten_min = 10 * 60 * fps #number of frames in 10 min to use as an index  

		filename_without_ext = os.path.splitext(os.path.basename(video))[0]

		f = h5py.File(h5_file, 'r')

		print("Keys: %s" % f.keys())

		#Get the keys from the file
		a_group_key = list(f.keys())[0]

		#convert the keys to a dataframe to make them easier to handle and to plot
		data = pd.read_hdf(h5_file, a_group_key)

		x = data.iloc[fps:ten_min, 0][data.iloc[:,2]>0.7] #10 min
		y = data.iloc[fps:ten_min, 1][data.iloc[:,2]>0.7]


		#create trajectory
		#you can control the outliers by modifying the likelihood value [data.iloc[:,2]>0.7]
		plt.figure(figsize=(4,4), dpi=300)
		plt.plot(x,y, color='k', linewidth=1)
		plt.axis('off')
		plt.gca().invert_yaxis() #otherwise the images appear mirror imaged
		plt.savefig('%s_Trajectory.png' % (filename_without_ext))
        trajectory = os.path.abspath(glob.glob('*_Trajectory.png')[0])
        return trajectory


draw_trajectory = Node(name = 'Draw_Trajectory',
                  interface = Function(input_names = ['h5_file','video'],
                  output_names = ['trajectory'],
                  function = draw_trajectory))

#-----------------------------------------------------------------------------------------------------
# In[8]:
#Draw Trajectory

def draw_density_map(h5_file, video):
		import os
		import h5py #top open h5 binary files
		import matplotlib.pyplot as plt
		import matplotlib
		import pandas as pd
		import numpy as np
		import cv2 
		import glob
		from scipy.stats import kde 
		np.set_printoptions(suppress=True) 

		#Get fps to use it for indexing, we need only 10 min from beg
		#so the indexing will be from 0:
		cap = cv2.VideoCapture(video)

		fps = int(cap.get(cv2.CAP_PROP_FPS))

		ten_min = 10 * 60 * fps #number of frames in 10 min to use as an index  

		filename_without_ext = os.path.splitext(os.path.basename(video))[0]

		f = h5py.File(h5_file, 'r')

		print("Keys: %s" % f.keys())

		#Get the keys from the file
		a_group_key = list(f.keys())[0]

		#convert the keys to a dataframe to make them easier to handle and to plot
		data = pd.read_hdf(h5_file, a_group_key)


		x = data.iloc[fps:ten_min, 0][data.iloc[:,2]>0.7] #10 min
		y = data.iloc[fps:ten_min, 1][data.iloc[:,2]>0.7]


		# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
		nbins=300 #300 is a very good compromise both computationally and aesthetically 
		k = kde.gaussian_kde([x,y])
		xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))

		plt.figure(figsize=(6,4), dpi=300)
		plt.pcolormesh(xi, yi, (zi.reshape(xi.shape) - zi.min())/(zi.max()), cmap='jet') #normalize zi value to get colorbar from 0-1
		plt.colorbar(ticks=[0,0.2,0.4,0.6,0.8,1])
		plt.axis('off')
		plt.gca().invert_yaxis()

		plt.savefig('%s_Density.png' % (filename_without_ext))
		density_map = os.path.abspath(glob.glob('*_Density.png')[0])
		return density_map

draw_density_map = Node(name = 'Draw_Density_Map',
                  interface = Function(input_names = ['h5_file','video'],
                  output_names = ['density_map'],
                  function = draw_density_map))


#-----------------------------------------------------------------------------------------------------
# In[9]:
#draw the compartments and extract the metrics
def get_metrics(h5_file, video):
		import os
		import h5py #top open h5 binary files
		import matplotlib.pyplot as plt
		import matplotlib
		import pandas as pd
		import numpy as np
		import cv2 
		import csv
		import glob

		np.set_printoptions(suppress=True) 

		#Get fps to use it for indexing, we need only 10 min from beg
		#so the indexing will be from 0:
		cap = cv2.VideoCapture(video)

		fps = int(cap.get(cv2.CAP_PROP_FPS))

		success,image = cap.read() #an image to use as a background 

		ten_min = 10 * 60 * fps #number of frames in 10 min to use as an index  

		filename_without_ext = os.path.splitext(os.path.basename(video))[0]

		f = h5py.File(h5_file, 'r')

		print("Keys: %s" % f.keys())

		#Get the keys from the file
		a_group_key = list(f.keys())[0]

		#convert the keys to a dataframe to make them easier to handle and to plot
		data = pd.read_hdf(h5_file, a_group_key)

		x = data.iloc[fps:ten_min, 0][data.iloc[:,2]>0.7] #10 min
		y = data.iloc[fps:ten_min, 1][data.iloc[:,2]>0.7]

		Z = np.column_stack((x,y))
		##############################################################################
		#Euclidean distance
		#we need to calculate the euclidean distance between each two consecutive frames
	    #so we make two matrices and remove the last frame from the first and the first frame from the second
	    #aka each two corresponding rows are two consecutive frames
	    #I tried to do it with euclidean distance in scipy, but it is more computationally expensive
	    #and gives exactly the same result
	    Z1 = Z[0:-1]  
	    Z2 = Z[1:, :]
	    
	    #the euclidean distance formula:  dist((x, y), (a, b)) = √(x - a)² + (y - b)²
	    diff = (Z1 - Z2)
	    
	    squared = diff**2
	    sumed = np.sum(squared, axis=1)
	    distances = np.sqrt(sumed) 
	    #from the calibration of the video, we know that 216pixels = 50cm aka 1cm=4.32pixel
	    total_distance = np.sum(distances) #pixels
	    total_distance = total_distance / 4.32 #cm
		


		################################################################################################################
		#Velocity
	    no_of_frames = Z.shape[0]
	    no_of_sec = no_of_frames / fps
	    velocity = total_distance / no_of_sec #velocity cm/sec
	    

	    #################################################################################################################
	    #get the mean
	    mean = np.mean([Z.max(axis=0), Z.min(axis=0)], axis=0) + 10 #10 is half arm width, fixed for my + maze arenas
	    
	    #draw a square with mean as the center point to delineate the center of the maze
	    shape1 = matplotlib.patches.RegularPolygon((mean[0],mean[1]),numVertices=4, radius=40, color='r', alpha=0.5)
	    
	    #get the 4 points that constitute the heads of the center square
	    square = shape1.get_verts() #they are 5 here to close the square
	    
	    #get the points inside of the center square
	    inside = shape1.contains_points(Z) #must contain Z, requires 2dim
	    outside = ~ inside #inrquality operator
	    
	    #determine the size and resolution of your figure
	    plt.figure(figsize=(8,6), dpi=300)
	 
	    #draw the trajectory
	    plt.plot(x,y, 'k-', linewidth=1, alpha=0.5)
	    plt.plot(x,y, 'b.',markersize=10, alpha=0.2) #I always do this as a quality control. If there is any blue points obvious
	                        #this means there is some kind of error that needs to be debugged
	    
	    #draw the frames inside and outside the center box
	    plt.plot(x[inside],y[inside],'w.')

	    plt.plot(x[outside],y[outside],'k.')

	    #plot the frames in each of the four arms
	    plt.plot(x[outside][x < mean[0]][y > mean[1]],
	         y[outside][x < mean[0]][y > mean[1]],'r.')


	    plt.plot(x[outside][x < mean[0] ][y < mean[1]],
	         y[outside][x < mean[0]][y < mean[1]],'m.')
	    
	    
	    plt.plot(x[outside][x > mean[0]][y < mean[1]],
	         y[outside][x > mean[0]][y < mean[1]],'y.')


	    plt.plot(x[outside][x > mean[0]][y > mean[1]],
	         y[outside][x > mean[0]][y > mean[1]],'g.', )

	    
	    #plot the square, last step to make it over the dots in order to be obvious
	    plt.plot(shape1.get_verts()[:,0],shape1.get_verts()[:,1], 'k--')
	    plt.plot(shape1.get_verts()[:,0],shape1.get_verts()[:,1], 'ko')
	    
	    #draw the mean last for the exact same reason
	    plt.plot(mean[0],mean[1], 'ko')
	    
	    #now plot your frame you extracted earlier, this will invert y axis aka put in the correct order
	    plt.imshow(image) #you do not need to invert y axis here, plotting an image forces it
	    
	    plt.axis('off')
	    plt.savefig('%s_Divisions.png' % (filename_without_ext))
	    divisions = os.path.abspath(glob.glob('*_Divisions.png')[0])
        
	    ######################################################################################################
	    
	    #use the number of frames to get the time spent in each arm
	    sec_in_center = len(x[inside]) / fps
	    
	    sec_in_arm1 = len(x[outside][x < mean[0]][y > mean[1]]) / fps #red arm
	    sec_in_arm2 = len(x[outside][x < mean[0]][y < mean[1]]) / fps #magenta  
	    sec_in_arm3 = len(x[outside][x > mean[0]][y < mean[1]]) / fps #yellow
	    sec_in_arm4 = len(x[outside][x > mean[0]][y > mean[1]]) / fps #green

	    time_in_center = sec_in_center
	    time_in_center_percentage = (time_in_center * fps) / len(x)
	    
	    time_in_opened_arms = sec_in_arm2 + sec_in_arm4
	    time_in_opened_arms_percentage = (time_in_opened_arms * fps)/ len(x) 
	    
	    
	    time_in_closed_arms = sec_in_arm1 + sec_in_arm3
	    time_in_closed_arms_percentage = (time_in_closed_arms * fps) / len(x)
	    
	    #open/close time ratio
	    opened_to_closed_ratio = time_in_opened_arms/time_in_closed_arms 
	    
	    ############################################################################################
	    #write to a csv file
	    import csv
	    
	    csvRow = [filename_without_ext, fps, total_distance, velocity, time_in_center, time_in_center_percentage, 
	             time_in_opened_arms, time_in_opened_arms_percentage,
	             time_in_closed_arms, time_in_closed_arms_percentage, opened_to_closed_ratio]


	    csvfile = filename_without_ext + 'metrics.csv'
	    
	    with open(csvfile, "a") as fp:
	        wr = csv.writer(fp, dialect='excel')
	        wr.writerow(['filename_without_ext', 'fps', 'total_distance', 'velocity', 'time_in_center', 'time_in_center_percentage', 
	             'time_in_opened_arms', 'time_in_opened_arms_percentage',
	             'time_in_closed_arms', 'time_in_closed_arms_percentage', 'opened_to_closed_ratio'])

	        wr.writerow(csvRow)

	    csv_output = os.path.abspath(glob.glob('*.csv')[0])

	    ############################################################################################
	    print ('Total time spent in the center: %0.4f seconds'%time_in_center)
	    print ('Percent of time spent in the center: %0.4f'%time_in_center_percentage)
	    
	    
	    print ('Total time spent in the opened arms: %0.4f seconds'%time_in_opened_arms)
	    print ('Percent of time spent in the opened arms: %0.4f '%time_in_opened_arms_percentage)
	    
	    print ('Total time spent in the opened arms: %0.4f seconds'%time_in_closed_arms)
	    print ('Percent of time spent in the closed arms: %0.4f '%time_in_closed_arms_percentage)
	    
	    print ('opened arms to closed arms ratio is: %0.4f' %opened_to_closed_ratio)

	    return divisions, csv_output

get_metrics = Node(name = 'Get_Metrics',
                  interface = Function(input_names = ['h5_file','video'],
                  output_names = ['divisions', 'csv_output'],
                  function = get_metrics))
#-----------------------------------------------------------------------------------------------------
# In[X]:

Plus_Maze_workflow.connect ([

      (infosource, selectfiles,[('subject_id','subject_id')]),

      (selectfiles, deeplabcut, [('plus_maze','video')]),

      # (deeplabcut, datasink, [('labeled_video','@labeled_video'), ('h5_file','@h5_file')]),

      (deeplabcut, draw_trajectory, [('h5_file','h5_file')]),

      (selectfiles, draw_trajectory, [('plus_maze','video')]),

      (draw_trajectory,datasink, [('trajectory','@trajectory')]),

      (deeplabcut, draw_density_map, [('h5_file','h5_file')]),

      (selectfiles, draw_density_map, [('plus_maze','video')]),

      (draw_density_map, datasink, [('density_map','@density_map')]),


      (deeplabcut, get_metrics, [('h5_file','h5_file')]),	

      (selectfiles, get_metrics, [('plus_maze','video')]),

      (get_metrics, datasink, [('divisions','@divisions'),('csv_output','@csv_output')]),

  ])


Plus_Maze_workflow.write_graph(graph2use='flat')
Plus_Maze_workflow.run('MultiProc', plugin_args={'n_procs': 16})

#-----------------------------------------------------------------------------------------------------
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

# subject_list = ['229', '230', '232', '233', 
#                 '234', '235', '237', '242', 
#                 '243', '244', '245', '252', 
#                 '253', '255', '261', '262', 
#                 '263', '264', '273', '274', 
#                 '281', '282', '286', '287', 
#                 '362', '363', '364', '365', 
#                 '366']

# subject_list = ['229', '230', '365', '274']
                
subject_list = ['230']


output_dir  = 'Open_Field_output'
working_dir = 'Open_Field_workingdir'

Open_Field_workflow = Workflow (name = 'Open_Field_workflow')
Open_Field_workflow.base_dir = opj(experiment_dir, working_dir)

#-----------------------------------------------------------------------------------------------------
# In[3]:


# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

#-----------------------------------------------------------------------------------------------------
# In[4]:

templates = {

             'plus_maze'  : 'Data/{subject_id}/open_field_{subject_id}.mp4'
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

        path_config = '/home/in/aeed/open_field/Open_Field-Amr-2019-01-04/config.yaml'


        deeplabcut.analyze_videos(path_config, [video], save_as_csv=True)

        deeplabcut.create_labeled_video(path_config, [video])


        h5_file = os.path.abspath(glob.glob('open_field_*.h5')[0])
        labeled_video = os.path.abspath(glob.glob('open_field_*_labeled.mp4')[0])

        pickle = os.path.abspath(glob.glob('open_field_*.pickle')[0])
        csv_file = os.path.abspath(glob.glob('open_field_*.csv')[0])

        #I failed to force deeplabcut to put the output in the workflow directory, so I am copying them
        #like a caveman

        subj_no = os.getcwd()[-3:]
        experiment_dir = '/home/in/aeed/Work/October_Acquistion/'
        working_dir = 'Open_Field_workingdir'

        copy2(h5_file, '%s%s/Open_Field_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))
        copy2(labeled_video,'%s%s/Open_Field_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))
        copy2(pickle,'%s%s/Open_Field_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))
        copy2(csv_file,'%s%s/Open_Field_workflow/_subject_id_%s/DeepLabCut/'%(experiment_dir, working_dir ,subj_no))

        return  h5_file, labeled_video


          


deeplabcut = Node(name = 'DeepLabCut',
                  interface = Function(input_names = ['video'],
                  output_names = ['h5_file', 'labeled_video'],
                  function = deeplabcut))
#-----------------------------------------------------------------------------------------------------
# In[6]:
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

        #Get fps to use it for indexing, we need only 30 min from beg
        #so the indexing will be from 0:
        cap = cv2.VideoCapture(video)

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        thirty_min = 30 * 60 * fps #number of frames in 30 min to use as an index  

        filename_without_ext = os.path.splitext(os.path.basename(video))[0]

        f = h5py.File(h5_file, 'r')

        print("Keys: %s" % f.keys())

        #Get the keys from the file
        a_group_key = list(f.keys())[0]

        #convert the keys to a dataframe to make them easier to handle and to plot
        data = pd.read_hdf(h5_file, a_group_key)

        x = data.iloc[fps:thirty_min, 0][data.iloc[:,2]>0.7] #10 min
        y = data.iloc[fps:thirty_min, 1][data.iloc[:,2]>0.7]


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

        #Get fps to use it for indexing, we need only 30 min from beg
        #so the indexing will be from 0:
        cap = cv2.VideoCapture(video)

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        thirty_min = 30 * 60 * fps #number of frames in 30 min to use as an index  

        filename_without_ext = os.path.splitext(os.path.basename(video))[0]

        f = h5py.File(h5_file, 'r')

        print("Keys: %s" % f.keys())

        #Get the keys from the file
        a_group_key = list(f.keys())[0]

        #convert the keys to a dataframe to make them easier to handle and to plot
        data = pd.read_hdf(h5_file, a_group_key)


        x = data.iloc[fps:thirty_min, 0][data.iloc[:,2]>0.7] #10 min
        y = data.iloc[fps:thirty_min, 1][data.iloc[:,2]>0.7]


        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data exthirtyts
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

        thirty_min = 30 * 60 * fps #number of frames in 30 min to use as an index  

        filename_without_ext = os.path.splitext(os.path.basename(video))[0]

        f = h5py.File(h5_file, 'r')

        print("Keys: %s" % f.keys())

        #Get the keys from the file
        a_group_key = list(f.keys())[0]

        #convert the keys to a dataframe to make them easier to handle and to plot
        data = pd.read_hdf(h5_file, a_group_key)

        x = data.iloc[fps:thirty_min, 0][data.iloc[:,2]>0.7] #10 min
        y = data.iloc[fps:thirty_min, 1][data.iloc[:,2]>0.7]

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

        plt.figure(figsize=(8,6), dpi=300)
        plt.plot(x,y, color='k', linewidth=1)
        plt.axis('off')
        #plot all the frames 
        plt.plot(x,y, 'b.')
        #the same as the one above, but connecting them together helps me to see 
        #if I am missing something
        plt.plot(x,y, 'b-', alpha=0.2) 

        #get the center point by averaging the extreme points from the edges of the box
        mean = np.mean([Z.max(axis=0), Z.min(axis=0)], axis=0)
        
        #plot the mean point as an indication I am in the correct way
        plt.plot(mean[0],mean[1], 'ko')
        
        #plot the two main diagonals
        plt.plot((Z.max(axis=0)[0], Z.min(axis=0)[0]), (Z.min(axis=0)[1], Z.max(axis=0)[1]),'r', lw=1.5) 
        plt.plot((Z.max(axis=0)[0], Z.min(axis=0)[0]), (Z.max(axis=0)[1], Z.min(axis=0)[1]),'r', lw=1.5) 
        

        #get the coordinates of the center of the field based on the center point
        square_x = [(Z.min(axis=0)[0] + mean[0]) / 2,
                    (Z.max(axis=0)[0] + mean[0]) / 2,
                    (Z.max(axis=0)[0] + mean[0]) / 2,
                    (Z.min(axis=0)[0] + mean[0]) / 2,
                    (Z.min(axis=0)[0] + mean[0]) / 2] #repeat last point to close the square



        square_y = [( Z.min(axis=0)[1] + mean[1]) / 2,
                    ( Z.min(axis=0)[1] + mean[1]) / 2,
                    ( Z.max(axis=0)[1] + mean[1]) / 2,
                    ( Z.max(axis=0)[1] + mean[1]) / 2,
                    ( Z.min(axis=0)[1] + mean[1]) / 2]#repeat last point to close the square

        #plot the center sqaure
        
        plt.plot((square_x), (square_y), 'k-')
        plt.plot((square_x), (square_y), 'ko')
        
        #plot all the frames that are inside the center square
        
        plt.plot(x[x > square_x[0]][x < square_x[1]][y > square_y[0]][y < square_y[2]],y[x > square_x[0]][x < square_x[1]][y > square_y[0]][y < square_y[2]], 'r.')

        
        #now get the coordinates of the boxes at each corner of the field
        rectangle1 = plt.Rectangle(((Z.min(axis=0)[0] + mean[0] )/ 2,( Z.min(axis=0)[1] + mean[1]) / 2), 50, 50, fc='g', angle=180, alpha=.7)

        rectangle2 = plt.Rectangle(((Z.max(axis=0)[0] + mean[0] )/ 2,( Z.min(axis=0)[1] + mean[1]) / 2), 50, 50, fc='m', angle=270, alpha=.7)

        rectangle3 = plt.Rectangle(((Z.max(axis=0)[0] + mean[0] )/ 2,( Z.max(axis=0)[1] + mean[1]) / 2), 50, 50, fc='y', angle=-360, alpha=.7)

        rectangle4 = plt.Rectangle(((Z.min(axis=0)[0] + mean[0] )/ 2,( Z.max(axis=0)[1] + mean[1]) / 2), 50, 50, fc='k', angle=90, alpha=.7)


        #plot the m with different colors
        plt.gca().add_patch(rectangle1)

        plt.gca().add_patch(rectangle2)

        plt.gca().add_patch(rectangle3)

        plt.gca().add_patch(rectangle4)



        rect1_xy = rectangle1.get_xy()
        rect2_xy = rectangle2.get_xy()
        rect3_xy = rectangle3.get_xy()
        rect4_xy = rectangle4.get_xy()


        #plot the frames inside each corner rectangle with the same color as the rectangle
        #to assess things visually
        plt.plot(x[x < rect1_xy[0]][y < rect1_xy[1]],y[x < rect1_xy[0]][y < rect1_xy[1]], 'g.')

        plt.plot(x[x > rect2_xy[0]][y < rect2_xy[1]],y[x > rect2_xy[0]][y < rect2_xy[1]], 'm.')

        plt.plot(x[x > rect3_xy[0]][y > rect3_xy[1]],y[x > rect3_xy[0]][y > rect3_xy[1]], 'y.')

        plt.plot(x[x < rect4_xy[0]][y > rect4_xy[1]],y[x < rect4_xy[0]][y > rect4_xy[1]], 'k.')

    #     plt.gca().invert_yaxis() #we do not need it anymore, since I am plotting a frame with plt.imshow
                                   #it takes care of putting y axis to real orientation
        
        #####################################################################################################
        #get a frame and put it as background

        cap = cv2.VideoCapture(video)
        success,image = cap.read()
        plt.imshow(image)
        
        plt.savefig('%s_Divisions.png' % (filename_without_ext))
        divisions = os.path.abspath(glob.glob('*_Divisions.png')[0])
        ######################################################################################################
        
        #use the number of frames to get the time spent in each corner
        sec_in_corner1 = len(x[x < rect1_xy[0]][y < rect1_xy[1]]) / fps
        sec_in_corner2 = len(x[x > rect2_xy[0]][y < rect2_xy[1]]) / fps
        sec_in_corner3 = len(x[x > rect3_xy[0]][y > rect3_xy[1]]) / fps
        sec_in_corner4 = len(x[x < rect4_xy[0]][y > rect4_xy[1]]) / fps
        
        #the same with the center
         
        sec_in_center = len(x[x > square_x[0]][x < square_x[1]][y > square_y[0]][y < square_y[2]]) / fps
        percent_in_center = (sec_in_center * fps) / len(x)
        
        #the same with the corners
        
        total_time_in_corners = sec_in_corner1 + sec_in_corner2 + sec_in_corner3 + sec_in_corner4
        percent_in_corners = (total_time_in_corners * fps) / len(x)
        #ratio
        center_corners_ratio = sec_in_center / total_time_in_corners


        ########################################################################################################
                #write to a csv file
        import csv
        
        csvRow = [filename_without_ext, fps, total_distance, velocity, sec_in_center, percent_in_center,
                   total_time_in_corners, percent_in_corners, center_corners_ratio]


        csvfile = filename_without_ext + 'metrics.csv'
        
        with open(csvfile, "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(['filename_without_ext', 'fps', 'total_distance', 'velocity', 'sec_in_center', 'percent_in_center',
                   'total_time_in_corners', 'percent_in_corners', 'center_corners_ratio'])

            wr.writerow(csvRow)

        csv_output = os.path.abspath(glob.glob('*.csv')[0]) 


        print ('Total time spent in the center: %0.4f seconds'%sec_in_center)
        print ('Percent time spent in the center: %0.4f'%percent_in_center)
        
        print ('Total time spent in the 4 corners: %0.4f seconds'%total_time_in_corners)
        print ('Percent time spent in the corners: %0.4f'%percent_in_corners)
        
        print ('Center to Corners ratio is: %0.4f' %center_corners_ratio)
        
        return divisions, csv_output

get_metrics = Node(name = 'Get_Metrics',
                  interface = Function(input_names = ['h5_file','video'],
                  output_names = ['divisions', 'csv_output'],
                  function = get_metrics))

#-----------------------------------------------------------------------------------------------------------------------------
#In[X]:

Open_Field_workflow.connect ([

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


Open_Field_workflow.write_graph(graph2use='flat')
Open_Field_workflow.run('MultiProc', plugin_args={'n_procs': 16})



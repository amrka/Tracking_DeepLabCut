
# coding: utf-8

# In[2]:


import deeplabcut
import glob
import os


# In[3]:


cd /home/amr/Trial_DeepLabCut


# In[4]:


pwd


# In[5]:


ls


# In[6]:


deeplabcut.create_new_project('Mouse', 'Amr', ['/home/amr/Trial_DeepLabCut/mouse.avi'])

# in case of multiple videos
videolist = glob.glob(os.path.abspath("*.mov"))


# # <font color=red>Now, You open the config.yaml file to modify the body part you want to track and if you want, you can modify the number of frames to pick</font>

# In[3]:


path_config = '/home/amr/Trial_DeepLabCut/Mouse-Amr-2018-12-03/config.yaml'


# In[4]:


deeplabcut.extract_frames(path_config, 'automatic', 'kmeans')


# In[9]:


deeplabcut.label_frames(path_config, Screens=2)


# Now, a GUI will open for you to label the body part you want

# In[10]:


deeplabcut.check_labels(path_config)


# In[11]:


deeplabcut.create_training_dataset(path_config)


# In[13]:


deeplabcut.train_network(path_config, saveiters='1000', displayiters='1')


# In[14]:


deeplabcut.evaluate_network(path_config)


# In[15]:


deeplabcut.analyze_videos(path_config, ['/home/amr/Trial_DeepLabCut/mouse.avi'], save_as_csv=True)


# In[26]:


deeplabcut.create_labeled_video(path_config, ['/home/amr/Trial_DeepLabCut/mouse.avi'])


# In[27]:


deeplabcut.create_labeled_video(path_config, ['/home/amr/Trial_DeepLabCut/mouse.avi'], save_frames=True)


# In[29]:


deeplabcut.plot_trajectories(path_config, ['/home/amr/Trial_DeepLabCut/mouse.avi'])


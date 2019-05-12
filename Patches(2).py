
# coding: utf-8

# In[8]:


import random
import os
#from skimage import io
import numpy as np
import glob
import SimpleITK as sitk
from keras.utils import np_utils


# In[3]:


path = '/home/bhakti/Downloads/Train data (1)'
listing = os.listdir(path)


# In[4]:


Path_of_Train= glob.glob("/home/bhakti/Downloads/Train data (1)/**")
len(Path_of_Train)


# In[6]:


Train = []
for i in range(len(Path_of_Train)):
        
        flair = glob.glob( Path_of_Train[i] + '/*_flair.nii.gz')
        t2 = glob.glob(Path_of_Train[i] + '/*_t2.nii.gz')
        gt = glob.glob(Path_of_Train[i] + '/*_seg.nii.gz')
        t1 = glob.glob(Path_of_Train[i] + '/*_t1.nii.gz')
        t1c = glob.glob( Path_of_Train[i] + '/*_t1ce.nii.gz')
        
        scans = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
        tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]
        Train.append(tmp)   
        


# In[46]:


# Add images in discontinues manner
tra_im_array = np.array(Train[17:24])


# In[47]:


len(tra_im_array)


# In[48]:


tra_im_array.shape


# In[49]:


# Taking middle slices only
tra_im_array=tra_im_array[:,:,14:141,40:200,40:200]


# In[50]:


tra_im_array.shape


# In[51]:


patches, labels = [], []
count = 0


# In[52]:


gt_im = np.swapaxes(tra_im_array, 0, 1)[4] 


# In[54]:


msk = np.swapaxes(tra_im_array, 0, 1)[0]


# In[55]:


tmp_shp = gt_im.shape


# In[56]:


gt_im = gt_im.reshape(-1).astype(np.uint8)
msk = msk.reshape(-1).astype(np.float32)


# In[57]:


indices = np.squeeze(np.argwhere((msk!=-9.0) & (msk!=0.0)))
del msk


# In[58]:


type(indices)


# In[59]:


np.random.shuffle(indices)


# In[60]:


gt_im = gt_im.reshape(tmp_shp)


# In[61]:


i = 0
d  = 4
h = 30
w = 30
num_patches = 146*20*3
pix = len(indices)
while (count<num_patches) and (pix>i):
    #randomly choose an index
    ind = indices[i]
    i+= 1
    #reshape ind to 3D index
    ind = np.unravel_index(ind, tmp_shp)
    # get the patient and the slice id
    patient_id = ind[0]
    slice_idx=ind[1]
    p = ind[2:]
    #construct the patch by defining the coordinates
    p_y = (p[0] - (h)/2, p[0] + (h)/2)
    p_x = (p[1] - (w)/2, p[1] + (w)/2)
    p_x=list(map(int,p_x))
    p_y=list(map(int,p_y))
    
    #take patches from all modalities and group them together
    tmp =tra_im_array[patient_id][0:4, slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]
    #take the coresponding label patch
    lbl=gt_im[patient_id,slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]

    #keep only paches that have the desired size
    if tmp.shape != (d, h, w) :
        continue
    patches.append(tmp)
    labels.append(lbl)
    count+=1
patches = np.array(patches)
labels=np.array(labels)
       


# In[66]:


#save to disk as npy files
np.save( "x3",patches )
np.save( "y3",labels)   


# In[24]:


# Save fractions in different variables
Y_labels_3=np.load("y3_dataset_first_part.npy").astype(np.uint8)
X_patches_3=np.load("x3_dataset_first_part.npy").astype(np.float32)
Y_labels_2=np.load("y2_dataset_first_part.npy").astype(np.uint8)
X_patches_2=np.load("x2_dataset_first_part.npy").astype(np.float32)
Y_labels_1=np.load("y1_dataset_first_part.npy").astype(np.uint8)
X_patches_1=np.load("x1_dataset_first_part.npy").astype(np.float32)#


# In[25]:


X_patches=np.concatenate((X_patches_1, X_patches_2,X_patches_3), axis=0)
Y_labels=np.concatenate((Y_labels_1, Y_labels_2,Y_labels_3), axis=0)
del Y_labels_2,X_patches_2,Y_labels_1,X_patches_1,Y_labels_3,X_patches_3


# In[41]:


# shuffle them
shuffle = list(zip(X_patches1, Y_labels1))
np.random.seed(138)
np.random.shuffle(shuffle)
X_patches1 = np.array([shuffle[i][0] for i in range(len(shuffle))])
Y_labels1 = np.array([shuffle[i][1] for i in range(len(shuffle))])
del shuffle


# In[16]:


Y_labels=np.load("y1.npy").astype(np.uint8)
X_patches=np.load("x1.npy").astype(np.float32)


# In[17]:


Y_labels1=np.load("y2.npy").astype(np.uint8)
X_patches1=np.load("x2.npy").astype(np.float32)


# In[18]:


Y_labels2=np.load("y3.npy").astype(np.uint8)
X_patches2=np.load("x3.npy").astype(np.float32)


# In[20]:


# Transpose them
X_patches2=np.transpose(X_patches2,(0,2,3,1)).astype(np.float32)


# In[21]:


X_patches1=np.transpose(X_patches1,(0,2,3,1)).astype(np.float32)


# In[22]:


X_patches=np.transpose(X_patches,(0,2,3,1)).astype(np.float32)




X_patchesF=np.concatenate((X_patches,X_patches_4,X_patches2,X_patches1), axis=0)



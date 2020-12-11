import os
import math
import glob

import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage.transform import resize,rotate 
from skimage.io import imread
from scipy.ndimage import zoom

INPUT_HEIGHT = 112
INPUT_WIDTH = 112
INPUT_DEPTH = 112
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_DEPTH = 224

def random_horizontal_flip(X,y,p=0.5):
    lottery = np.random.random_sample()
    if lottery>p:
        return X[:,:,::-1], y[:,:,::-1]
    else:
        return X,y

def random_rotate(X,y,rotate_degree=(-45,45)):
    degree = np.random.randint(rotate_degree[0],rotate_degree[1])

    X_rotated = np.zeros_like(X)
    y_rotated = np.zeros_like(y)

    for depth in range(X.shape[2]):
        X_rotated[:,:,depth] = rotate(X[:,:,depth],degree)
        y_rotated[:,:,depth] = rotate(y[:,:,depth],degree)

    return X_rotated, y_rotated

class PatchesDataset(keras.utils.Sequence):
    def __init__(self, batch_size, input_paths, target_paths):
        self.batch_size = batch_size
        self.input_paths = input_paths
        self.target_paths = target_paths
        
    def __len__(self):
        return len(self.target_paths) // self.batch_size
    
    def __getitem__(self,idx):
        i = idx * self.batch_size
        batch_input_paths = self.input_paths[i:i+self.batch_size]
        batch_target_paths = self.target_paths[i:i+self.batch_size]
        
        x = []
        y = []
        for j, (input_path,target_path) in enumerate(zip(batch_input_paths,batch_target_paths)):
            img = np.load(input_path)
            # img = img.reshape(img.shape+(1,))

            img_target = np.load(target_path)
            # img_target = img_target.reshape(img_target.shape+(1,))

            img,img_target = random_horizontal_flip(img,img_target)

            x.append(img)
            y.append(img_target)

        x = np.stack(x)
        x = tf.convert_to_tensor(x,dtype=tf.float32)
        x = x/255.0

        y = np.stack(y)
        y = tf.convert_to_tensor(y,dtype=tf.float32)
        y = y/255.0
        
        return x,y

class PatchesResizeDataset(keras.utils.Sequence):
    def __init__(self, batch_size, input_paths, target_paths):
        self.batch_size = batch_size
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.img_size = 112
        
    def __len__(self):
        return len(self.target_paths) // self.batch_size
    
    def __getitem__(self,idx):
        i = idx * self.batch_size
        batch_input_paths = self.input_paths[i:i+self.batch_size]
        batch_target_paths = self.target_paths[i:i+self.batch_size]
        
        x = []
        y = []
        for j, (input_path,target_path) in enumerate(zip(batch_input_paths,batch_target_paths)):
            img = np.load(input_path)
            # img = img.reshape(img.shape+(1,))

            img_target = np.load(target_path)
            # img_target = img_target.reshape(img_target.shape+(1,))

            img,img_target = random_horizontal_flip(img,img_target)

            x.append(img)
            y.append(img_target)

        x = np.stack(x)
        x = tf.convert_to_tensor(x,dtype=tf.float32)
        x = x/255.0

        x = tf.transpose(x,[0,3,1,2,4])
        x = tf.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
        x = tf.image.resize(x,[self.img_size,self.img_size])
        x = tf.reshape(x,[self.batch_size,x.shape[0]//self.batch_size,x.shape[1],x.shape[2],x.shape[3]])
        x = tf.transpose(x,[0,2,3,1,4])
        
        x = tf.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
        x = tf.image.resize(x,[x.shape[1],self.img_size])
        x = tf.reshape(x,[self.batch_size,x.shape[0]//self.batch_size,x.shape[1],x.shape[2],x.shape[3]])

        y = np.stack(y)
        y = tf.convert_to_tensor(y,dtype=tf.float32)
        y = y/255.0

        y = tf.transpose(y,[0,3,1,2,4])
        y = tf.reshape(y,[y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4]])
        y = tf.image.resize(y,[self.img_size,self.img_size],method="nearest")
        y = tf.reshape(y,[self.batch_size,y.shape[0]//self.batch_size,y.shape[1],y.shape[2],y.shape[3]])
        y = tf.transpose(y,[0,2,3,1,4])
        
        y = tf.reshape(y,[y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4]])
        y = tf.image.resize(y,[y.shape[1],self.img_size],method="nearest")
        y = tf.reshape(y,[self.batch_size,y.shape[0]//self.batch_size,y.shape[1],y.shape[2],y.shape[3]])
        
        return x,y

class InstantJpegDataset(keras.utils.Sequence):
    def __init__(self, batch_size, input_paths, target_paths, patch_shape, overlap, shuffle=True):

        self.batch_size = batch_size
        self.input_paths = sorted(glob.glob(os.path.join(input_paths,"*.jpg")))
        self.target_paths = sorted(glob.glob(os.path.join(target_paths,"*.bmp")))

        self.input_image_shape = self._calc_dim()

        self.patch_shape = patch_shape
        self.overlap = overlap

        self.shuffle = shuffle

        self.i_limit = math.ceil((self.input_image_shape[0]-patch_shape[0])/(patch_shape[0]-overlap[0])+1)
        self.j_limit = math.ceil((self.input_image_shape[1]-patch_shape[1])/(patch_shape[1]-overlap[1])+1)
        self.k_limit = math.ceil((self.input_image_shape[2]-patch_shape[2])/(patch_shape[2]-overlap[2])+1)

        self.num_patches = self.i_limit*self.j_limit*self.k_limit

        self.indexes = np.arange(self.num_patches)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _calc_dim(self):
        im_k = len(self.input_paths)
        im_sample = imread(self.input_paths[0])

        im_i = im_sample.shape[0]
        im_j = im_sample.shape[1]
        return (im_i,im_j,im_k)

    def _calc_slice(self,patch_number):
        i_now = patch_number// (self.j_limit*self.k_limit)
        j_now = (patch_number - (i_now*self.j_limit*self.k_limit)) // self.k_limit
        k_now = (patch_number - (i_now*self.j_limit*self.k_limit) - j_now*self.k_limit) 

        if i_now == self.i_limit-1:
            i_end = self.input_image_shape[0]
            i_start = i_end-self.patch_shape[0]
        else:
            i_start=i_now*self.overlap[0]
            i_end = i_start + self.patch_shape[0]

        if j_now == self.j_limit-1:
            j_end = self.input_image_shape[1]
            j_start = j_end-self.patch_shape[1]
        else:
            j_start=j_now*self.overlap[1]
            j_end = j_start + self.patch_shape[1]

        if k_now == self.k_limit-1:
            k_end = self.input_image_shape[2]
            k_start = k_end-self.patch_shape[2]
        else:
            k_start=k_now*self.overlap[2]
            k_end = k_start + self.patch_shape[2]

        return (i_start,i_end,j_start,j_end,k_start,k_end)
        
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def __getitem__(self,idx):
        i = idx * self.batch_size
        batch_patches = self.indexes[i:i+self.batch_size]

        x = []
        y = []
        for j, patch_number in enumerate(batch_patches):
            slice_index = self._calc_slice(patch_number)

            img_stack = []
            img_target_stack = []
            for idx_k in range(slice_index[4],slice_index[5]):
                img_slice = imread(self.input_paths[idx_k],as_gray=True)
                img_slice = img_slice[slice_index[0]:slice_index[1],slice_index[2]:slice_index[3]]
                img_stack.append(img_slice)

                img_target_slice = imread(self.target_paths[idx_k],as_gray=True)
                img_target_slice = img_target_slice[slice_index[0]:slice_index[1],slice_index[2]:slice_index[3]]
                img_target_stack.append(img_target_slice)

                # print(self.input_paths[idx_k])
                # print("img",img_slice.shape)
                # print(self.target_paths[idx_k])
                # print("img_target",img_target_slice.shape)

            img = np.stack(img_stack,axis=-1)
            img = img.reshape(img.shape+(1,))

            img_target = np.stack(img_target_stack, axis=-1)
            img_target = img_target.reshape(img_target.shape+(1,))

            img,img_target = random_horizontal_flip(img,img_target)
            # img,img_target = random_rotate(img,img_target)
            
            x.append(img)
            y.append(img_target)

        x = np.stack(x)
        x = tf.convert_to_tensor(x,dtype=tf.float32)
        x = x/255.0
        
        y = np.stack(y)
        y = tf.convert_to_tensor(y,dtype=tf.float32)
        y = y/255.0

        if IMG_HEIGHT != INPUT_HEIGHT or IMG_WIDTH != INPUT_WIDTH or IMG_DEPTH != INPUT_DEPTH:

            x = tf.transpose(x,[0,3,1,2,4])
            x = tf.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
            x = tf.image.resize(x,[INPUT_HEIGHT,INPUT_WIDTH])
            x = tf.reshape(x,[self.batch_size,x.shape[0]//self.batch_size,x.shape[1],x.shape[2],x.shape[3]])
            x = tf.transpose(x,[0,2,3,1,4])
            
            x = tf.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
            x = tf.image.resize(x,[x.shape[1],INPUT_DEPTH])
            x = tf.reshape(x,[self.batch_size,x.shape[0]//self.batch_size,x.shape[1],x.shape[2],x.shape[3]])

            y = tf.transpose(y,[0,3,1,2,4])
            y = tf.reshape(y,[y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4]])
            y = tf.image.resize(y,[INPUT_HEIGHT,INPUT_WIDTH],method="nearest")
            y = tf.reshape(y,[self.batch_size,y.shape[0]//self.batch_size,y.shape[1],y.shape[2],y.shape[3]])
            y = tf.transpose(y,[0,2,3,1,4])
            
            y = tf.reshape(y,[y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4]])
            y = tf.image.resize(y,[y.shape[1],INPUT_DEPTH],method="nearest")
            y = tf.reshape(y,[self.batch_size,y.shape[0]//self.batch_size,y.shape[1],y.shape[2],y.shape[3]])
        
        return x,y

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_patches)
        if self.shuffle:
            np.random.shuffle(self.indexes)

class InstantPatchesDataset(keras.utils.Sequence):
    def __init__(self, batch_size, input_paths, target_paths, patch_shape, overlap, shuffle=True):

        self.batch_size = batch_size
        # self.input_paths = input_paths
        # self.target_paths = target_paths

        self.input_image = np.load(input_paths)
        self.target_image = np.load(target_paths)

        self.patch_shape = patch_shape
        self.overlap = overlap

        self.shuffle = shuffle

        self.i_limit = math.ceil((self.input_image.shape[0]-patch_shape[0])/(patch_shape[0]-overlap[0])+1)
        self.j_limit = math.ceil((self.input_image.shape[1]-patch_shape[1])/(patch_shape[1]-overlap[1])+1)
        self.k_limit = math.ceil((self.input_image.shape[2]-patch_shape[2])/(patch_shape[2]-overlap[2])+1)

        self.num_patches = self.i_limit*self.j_limit*self.k_limit

        self.indexes = np.arange(self.num_patches)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _calc_slice(self,patch_number):
        i_now = patch_number// (self.j_limit*self.k_limit)
        j_now = (patch_number - (i_now*self.j_limit*self.k_limit)) // self.k_limit
        k_now = (patch_number - (i_now*self.j_limit*self.k_limit) - j_now*self.k_limit) 

        if i_now == self.i_limit-1:
            i_end = self.input_image.shape[0]
            i_start = i_end-self.patch_shape[0]
        else:
            i_start=i_now*self.overlap[0]
            i_end = i_start + self.patch_shape[0]

        if j_now == self.j_limit-1:
            j_end = self.input_image.shape[1]
            j_start = j_end-self.patch_shape[1]
        else:
            j_start=j_now*self.overlap[1]
            j_end = j_start + self.patch_shape[1]

        if k_now == self.k_limit-1:
            k_end = self.input_image.shape[2]
            k_start = k_end-self.patch_shape[2]
        else:
            k_start=k_now*self.overlap[2]
            k_end = k_start + self.patch_shape[2]

        return (i_start,i_end,j_start,j_end,k_start,k_end)
        
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def __getitem__(self,idx):
        i = idx * self.batch_size
        batch_patches = self.indexes[i:i+self.batch_size]

        x = []
        y = []
        for j, patch_number in enumerate(batch_patches):
            slice_index = self._calc_slice(patch_number)
            img = self.input_image[slice_index[0]:slice_index[1],slice_index[2]:slice_index[3],slice_index[4]:slice_index[5]]
            img = img.reshape(img.shape+(1,))

            img_target = self.target_image[slice_index[0]:slice_index[1],slice_index[2]:slice_index[3],slice_index[4]:slice_index[5]]
            img_target = img_target.reshape(img_target.shape+(1,))

            img,img_target = random_horizontal_flip(img,img_target)
            # img,img_target = random_rotate(img,img_target)
            
            x.append(img)
            y.append(img_target)

        x = np.stack(x)
        x = tf.convert_to_tensor(x,dtype=tf.float32)
        x = x/255.0
        
        y = np.stack(y)
        y = tf.convert_to_tensor(y,dtype=tf.float32)
        y = y/255.0

        if IMG_HEIGHT != INPUT_HEIGHT or IMG_WIDTH != INPUT_WIDTH or IMG_DEPTH != INPUT_DEPTH:

            x = tf.transpose(x,[0,3,1,2,4])
            x = tf.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
            x = tf.image.resize(x,[INPUT_HEIGHT,INPUT_WIDTH])
            x = tf.reshape(x,[self.batch_size,x.shape[0]//self.batch_size,x.shape[1],x.shape[2],x.shape[3]])
            x = tf.transpose(x,[0,2,3,1,4])
            
            x = tf.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
            x = tf.image.resize(x,[x.shape[1],INPUT_DEPTH])
            x = tf.reshape(x,[self.batch_size,x.shape[0]//self.batch_size,x.shape[1],x.shape[2],x.shape[3]])

            y = tf.transpose(y,[0,3,1,2,4])
            y = tf.reshape(y,[y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4]])
            y = tf.image.resize(y,[INPUT_HEIGHT,INPUT_WIDTH])
            y = tf.reshape(y,[self.batch_size,y.shape[0]//self.batch_size,y.shape[1],y.shape[2],y.shape[3]])
            y = tf.transpose(y,[0,2,3,1,4])
            
            y = tf.reshape(y,[y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4]])
            y = tf.image.resize(y,[y.shape[1],INPUT_DEPTH])
            y = tf.reshape(y,[self.batch_size,y.shape[0]//self.batch_size,y.shape[1],y.shape[2],y.shape[3]])
        
        return x,y
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.num_patches)
        if self.shuffle:
            np.random.shuffle(self.indexes)

class InstantPreresizedDataset(keras.utils.Sequence):
    def __init__(self, batch_size, input_paths, target_paths, patch_shape, overlap, shuffle=True):

        self.batch_size = batch_size
        # self.input_paths = input_paths
        # self.target_paths = target_paths

        original_input_image = np.load(input_paths)
        original_target_image = np.load(target_paths)

        self.patch_shape = [x for x in patch_shape]
        self.overlap = [x for x in overlap]

        zoom_ratio = (INPUT_HEIGHT / self.patch_shape[0],\
                        INPUT_WIDTH / self.patch_shape[1],\
                        INPUT_DEPTH / self.patch_shape[2])

        self.input_image = zoom(original_input_image,zoom_ratio)
        self.target_image = zoom(original_target_image,zoom_ratio)

        self.shuffle = shuffle

        for i,dim_ratio in enumerate(zoom_ratio):
            self.overlap[i] = math.ceil(self.overlap[i]*dim_ratio)

        self.patch_shape[0] = INPUT_HEIGHT
        self.patch_shape[1] = INPUT_WIDTH
        self.patch_shape[2] = INPUT_DEPTH

        self.i_limit = math.ceil((self.input_image.shape[0]-self.patch_shape[0])/(self.patch_shape[0]-self.overlap[0])+1)
        self.j_limit = math.ceil((self.input_image.shape[1]-self.patch_shape[1])/(self.patch_shape[1]-self.overlap[1])+1)
        self.k_limit = math.ceil((self.input_image.shape[2]-self.patch_shape[2])/(self.patch_shape[2]-self.overlap[2])+1)

        self.num_patches = self.i_limit*self.j_limit*self.k_limit

        self.indexes = np.arange(self.num_patches)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _calc_slice(self,patch_number):
        i_now = patch_number// (self.j_limit*self.k_limit)
        j_now = (patch_number - (i_now*self.j_limit*self.k_limit)) // self.k_limit
        k_now = (patch_number - (i_now*self.j_limit*self.k_limit) - j_now*self.k_limit) 

        if i_now == self.i_limit-1:
            i_end = self.input_image.shape[0]
            i_start = i_end-self.patch_shape[0]
        else:
            i_start=i_now*self.overlap[0]
            i_end = i_start + self.patch_shape[0]

        if j_now == self.j_limit-1:
            j_end = self.input_image.shape[1]
            j_start = j_end-self.patch_shape[1]
        else:
            j_start=j_now*self.overlap[1]
            j_end = j_start + self.patch_shape[1]

        if k_now == self.k_limit-1:
            k_end = self.input_image.shape[2]
            k_start = k_end-self.patch_shape[2]
        else:
            k_start=k_now*self.overlap[2]
            k_end = k_start + self.patch_shape[2]

        return (i_start,i_end,j_start,j_end,k_start,k_end)
        
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def __getitem__(self,idx):
        i = idx * self.batch_size
        batch_patches = self.indexes[i:i+self.batch_size]

        x = []
        y = []
        for j, patch_number in enumerate(batch_patches):
            slice_index = self._calc_slice(patch_number)
            img = self.input_image[slice_index[0]:slice_index[1],slice_index[2]:slice_index[3],slice_index[4]:slice_index[5]]
            img = img.reshape(img.shape+(1,))

            img_target = self.target_image[slice_index[0]:slice_index[1],slice_index[2]:slice_index[3],slice_index[4]:slice_index[5]]
            img_target = img_target.reshape(img_target.shape+(1,))

            img,img_target = random_horizontal_flip(img,img_target)
            # img,img_target = random_rotate(img,img_target)
            
            x.append(img)
            y.append(img_target)

        x = np.stack(x)
        x = tf.convert_to_tensor(x,dtype=tf.float32)
        x = x/255.0
        
        y = np.stack(y)
        y = tf.convert_to_tensor(y,dtype=tf.float32)
        y = y/255.0
        
        return x,y
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.num_patches)
        if self.shuffle:
            np.random.shuffle(self.indexes)

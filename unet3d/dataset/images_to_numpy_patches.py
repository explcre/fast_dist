import numpy as np
import os
import argparse
import glob
from tqdm import tqdm
import math
from skimage.transform import resize,rotate 
from skimage import img_as_ubyte
from skimage.io import imread
import pathlib

parser = argparse.ArgumentParser(description='3D Images to patches',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-s', '--src', type=str, default='./kitech',
                    help='Source Path')

parser.add_argument('-d', '--dest', type=str, default='./kitech_patches',
                    help='Target Path')

parser.add_argument('--patch_size',type=int, default=224)
parser.add_argument('--overlap_size',type=int, default=112)
parser.add_argument('--resize',type=int, default=None)

args = parser.parse_args()

def img_to_numpy(src_path):
    '''path contains image files s.t 0000.jpg to ~.jpg and label folder contatins 0000.bmp to ~.bmp'''

    imgs = sorted(glob.glob(os.path.join(src_path,"*.jpg")))
    labels = sorted(glob.glob(os.path.join(src_path,'label',"*.bmp")))

    img_stack = []
    for img in imgs:
        img_stack.append(imread(img,as_gray=True))

    np_img = np.stack(img_stack,axis=-1)
    np_img = np_img.reshape(np_img.shape+(1,))

    label_stack = []
    for label in labels:
        label_stack.append(imread(label,as_gray=True))

    np_label = np.stack(label_stack,axis=-1)
    np_label = np_label.reshape(np_label.shape+(1,))

    return np_img,np_label

def save_numpy_to_patches(numpy_img,numpy_label,origin,dest,patch_size,overlap_size,input_size):
    patch_slices = calc_patch_coord(numpy_img.shape,patch_size,overlap_size)
    for i,patch_coord in tqdm(enumerate(patch_slices),total=len(patch_slices)):
        img_slice = numpy_img[patch_coord[0]:patch_coord[1],patch_coord[2]:patch_coord[3],patch_coord[4]:patch_coord[5]]
        if input_size[0] != None:
            img_slice = img_as_ubyte(resize(img_slice,input_size))

        label_slice = numpy_label[patch_coord[0]:patch_coord[1],patch_coord[2]:patch_coord[3],patch_coord[4]:patch_coord[5]]
        if input_size[0] != None:
            label_slice = resize(label_slice,input_size,preserve_range=True,order=0)

        img_name = f'{origin[1]}_{origin[2]}_{i}'
        label_name = f'{origin[1]}_{origin[2]}_label_{i}'

        np.save(os.path.join(dest,img_name),img_slice)
        np.save(os.path.join(dest,label_name),label_slice)


def calc_patch_coord(tensor_shape,patch_size=(112,112,112),overlap=(56,56,56)):
    return_coord = []
    
    for i in range(math.ceil((tensor_shape[0]-patch_size[0])/(patch_size[0]-overlap[0])+1)):
        for j in range(math.ceil((tensor_shape[1]-patch_size[1])/(patch_size[1]-overlap[1])+1)):
            for k in range(math.ceil((tensor_shape[2]-patch_size[2])/(patch_size[2]-overlap[2])+1)):
                i_start = i*overlap[0]
                i_end = i*overlap[0]+patch_size[0]
                if i_end > tensor_shape[0]:
                    diff = i_end - tensor_shape[0]
                    i_end = tensor_shape[0]
                    i_start = i_start - diff
                    
                j_start = j*overlap[1]
                j_end = j*overlap[1]+patch_size[1]
                if j_end > tensor_shape[1]:
                    diff = j_end - tensor_shape[1]
                    j_end = tensor_shape[1]
                    j_start = j_start - diff
                    
                k_start = k*overlap[2]
                k_end = k*overlap[2]+patch_size[2]
                if k_end > tensor_shape[2]:
                    diff = k_end - tensor_shape[2]
                    k_end = tensor_shape[2]
                    k_start = k_start - diff
                    
                return_coord.append([i_start,i_end,j_start,j_end,k_start,k_end])
    return return_coord

if __name__ == '__main__':

    TRAIN_DEST_PATH = os.path.join(args.dest,'train')
    TEST_DEST_PATH = os.path.join(args.dest,'test')
    PATCH_SIZE = (args.patch_size,args.patch_size,args.patch_size)
    OVERLAP_SIZE = (args.overlap_size,args.overlap_size,args.overlap_size)
    INPUT_SIZE = (args.resize,args.resize,args.resize)

    pathlib.Path(TRAIN_DEST_PATH).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(TEST_DEST_PATH).mkdir(parents=True, exist_ok=True) 

    for dirs in os.listdir(args.src):
        if os.path.isdir(os.path.join(args.src,dirs)):
            for subdirs in os.listdir(os.path.join(args.src,dirs)):
                print(os.path.join(args.src,dirs,subdirs))
                if dirs == 'p2':
                    dest_path = TRAIN_DEST_PATH
                if dirs == 'p7':
                    dest_path = TEST_DEST_PATH
                np_img,np_label = img_to_numpy(os.path.join(args.src,dirs,subdirs))
                save_numpy_to_patches(np_img,np_label,
                                        origin=(args.src,dirs,subdirs),
                                        dest=dest_path,
                                        patch_size=PATCH_SIZE,
                                        overlap_size=OVERLAP_SIZE,
                                        input_size=INPUT_SIZE)

# 3. 3D-Unet README

### **To reproduce the examples**

1. Make Dataset
Go to /fast_dist/unet3d/dataset/
Unzip kitech.zip
Run make_patches.sh and make_resized_patches.sh
2. Execute Script
Go to /fast_dist/unet3d/

    JPEG Dataset : jpeg_example.sh

    JPEG pre-decoded : patche_example.sh

    JPEG pre-decoded + pre-resize : resized_patch_example.sh

### **To execute the code**

Tensorflow Default

python unet3d.py

HOROVOD Default

horovodrun -np 4 -H [localhost:4](http://localhost:4) python unet3d_horovod.py

### Arguments (can find with -h,—help option)

batch per gpu : -b or —batch (default = 1)

patch size : —patch-size (default = 224)

overlap size : —overlap-size (default = 112)

input size : —input-size (default = 112)

epochs : -e or —epochs (default = 10)

data path : —data-path (default = None)

log directory : —logdir (default = "./logs")

verbose : —verbose (default = 1)

Mixed precision : —fp16

choose jpeg dataset : —jpeg-dataset

choose pre-decoded dataset : —patch-dataset

choose pre-decoded + pre-resized dataset : —patch-resized-dataset (default)

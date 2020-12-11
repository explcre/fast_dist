#KITECH README

# 2. EfficientNet README

### **To reproduce the examples**

Go to /fast_dist/effi_imagenet/

Keras Default Dataset : keras_example.sh

TF Dataset : tf_example.sh

TF Dataset + prefetch : tf_pre_example.sh

HOROVOD + TF Dataset + prefetch : horovod_tf_pre_example.sh

HOROVOD + TF Dataset + prefetch + mixed precision : horovod_tf_pre_fp16_example.sh

### **To execute the code**

Tensorflow Default

python effi_imagenet.py

HOROVOD Default

horovodrun -np 4 -H [localhost:4](http://localhost:4) python effi_imagenet_horovod.py

### Arguments (can find with -h,—help option)

batch per gpu : -b or —batch (default = 64)

image size : —image-size (default = 224)

epochs : -e or —epochs (default = 10)

imagenet data path : —data-path (default = "/image")

checkpoint directory : —checkpoint-dir (default = "./training_checkpoints")

log directory : —logdir (default = "./logs")

verbose : —verbose (default = 1)

no prefetch : —no-prefetch

no tf dataset : —no-tf-dataset

enable mixed precision : —fp16

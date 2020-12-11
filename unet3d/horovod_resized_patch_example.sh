#! /bin/bash

horovodrun -np 4 -H localhost:4 python unet3d_horovod.py
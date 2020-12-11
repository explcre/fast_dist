#! /bin/bash

horovodrun -np 4 -H localhost:4 python effi_imagenet_horovod.py
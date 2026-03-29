##!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --reg_coef 1e-5 --steps 30000 --dim 2 --side 20 --batch_size 32 --dir results
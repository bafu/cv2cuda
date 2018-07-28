#!/bin/bash

rm -rf \
    build/ \
    cv2cuda/__pycache__/ \
    cv2cuda/gpuwrapper.cpp \
    cv2cuda/gpuwrapper.cpython-35m-x86_64-linux-gnu.so \
    preview.jpg \
    resize.jpg \
    warped.jpg \
    warped_gpu.jpg

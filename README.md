# OpenCV CUDA Wrapper Example

The example is based on the two Stack Overflow threads:

* [Accessing OpenCV CUDA Functions from Python (No PyCUDA)
](https://stackoverflow.com/questions/42125084)
* [OpenCV: Understanding warpPerspective / perspective transform](https://stackoverflow.com/questions/45717277)

Note: Most codes are wrote by [ostrumvulpes](https://stackoverflow.com/users/7292122/ostrumvulpes) and [stefan-at-wpf](https://stackoverflow.com/users/298288/stefan-at-wpf)

# How to Run Example

Precondition: OpenCV w/ CUDA support has been installed.

```
$ ./build.sh
$ python3 test.py
```

Then check `warped.jpg` and `warped_gpu.jpg`.

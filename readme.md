# TensorFlow Pose Realtime

This a openpose version which using a lightweighted backend of MobileNetV2 with some modifications.
It can runs almost **50fps** on GTX1080 - A realtime version of pose estimation. Here you can simply
run:

```
# some dependencies incase you are not installed
sudo pip3 install alfred-py
sudo pip3 install slidingwindow
# to pafprocess and run make.sh

python3 demo.py
```

And you will got this result:

<p align="center">
  <img src="https://s2.ax1x.com/2019/03/26/ANOJRf.gif">
</p>



The model weight already included.

## Training

For further training, you can access our **zhihu** for details, click
[here](http://zhuanlan.zhihu.com/ai-man), We will opensource the whole codes later as well as
training on hands joint estimation. Just stay tuned.


## Reference

this work based on original implementation of `ildoonet`,  thanks for original author excellent
work.. We made some changes to original code which contains whole structure and some training codes.

## License

this work release under Apache 2.0 License.

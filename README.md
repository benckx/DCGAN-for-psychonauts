# About

Use DCGAN to create trippy videos.

<a href="https://www.vjloops.com/stock-video/the-beautiful-people-21-139715.html">![](https://storage.googleapis.com/vjloops/139715_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/the-beautiful-people-114-139683.html">![](https://storage.googleapis.com/vjloops/139683_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/the-beautiful-people-112-139681.html">![](https://storage.googleapis.com/vjloops/139681_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/the-beautiful-people-23-139717.html">![](https://storage.googleapis.com/vjloops/139717_thumb0.jpg)</a>

<a href="https://www.vjloops.com/stock-video/elusive-features-loop-1-139232.html">![](https://storage.googleapis.com/vjloops/139232_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/elusive-features-125-139202.html">![](https://storage.googleapis.com/vjloops/139202_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/elusive-features-83-139155.html">![](https://storage.googleapis.com/vjloops/139155_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/elusive-features-95-139162.html">![](https://storage.googleapis.com/vjloops/139162_thumb0.jpg)</a>

<a href="https://www.vjloops.com/stock-video/fresh-and-simple-7-loop-3-140717.html">![](https://storage.googleapis.com/vjloops/140717_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/fresh-and-simple-8-loop-3-140747.html">![](https://storage.googleapis.com/vjloops/140747_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/fresh-and-simple-4-loop-3-140635.html">![](https://storage.googleapis.com/vjloops/140635_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/fresh-and-simple-5c-140711.html">![](https://storage.googleapis.com/vjloops/140711_thumb0.jpg)</a>

<a href="https://www.vjloops.com/stock-video/transmission-error-loop-13-143002.html">![](https://storage.googleapis.com/vjloops/143002_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/8i-bits-loop-1-143179.html">![](https://storage.googleapis.com/vjloops/143179_thumb0.jpg)</a>
<a href="https://www.vjloops.com//stock-video/8i-bits-loop-2-143180.html">![](https://storage.googleapis.com/vjloops/143180_thumb0.jpg)</a>
<a href="https://www.vjloops.com/stock-video/8i-bits-loop-9-143296.html">![](https://storage.googleapis.com/vjloops/143296_thumb0.jpg)</a>

More examples on [my portfolio](https://www.vjloops.com/index.php?portfolio=1&user=20585&svideo=1&items=500&str=1&showmenu=0)

## DCGAN

Deep Convolutional Generative Adversarial Networks (DCGAN) are used to generate realistic images. But if you sample 
the state of the model at every step and render them into a video, you can create a sort of "timelapse"
of the training process. This is what this project does.

The model is a fork of [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow). This
project is a generalization of the model, to allow more tinkering of the parameters, which can result in maybe 
less realistic but more visually interesting renders.

# Usage

    python3 multitrain.py --config my_config.csv --disable_cache 
    
You can use `nohup` if you want to be able to close the terminal:
   
    nohup python3 multitrain.py --config my_config.csv --disable_cache > log.out 2>&1&

There is also a bash script shortcut:

    ./run.sh my_config.csv 0,1

Which is equivalent to:

    nohup python3 multitrain.py --config my_config.csv --gpu_idx 0,1 --disable_cache > my_config.csv.out 2>&1&

Command parameters:

* `--gpu_idx 2` to pick a specific GPU device if you have several. You can also pick multiple GPU, 
the model will be spread on them: `--gpu_idx 0,1,2`. I would recommend to run the job on 1 GPU when possible
(to avoid communication overheads). If not possible, pick 2 or 3 GPU.
* `--disable_cache` disable caching of np data (you should add that if you use a lot of large images)
(I'll probably make this the default setting in the future)

## CSV config file

The CSV config file contains a list of jobs with different model and video parameters.
A minimal config CSV file must contain the following columns:

|name          |dataset       |grid_width|grid_height|video_length|
|---           |---           |---       |---        |---         |
|job01         |images_folder |3         |3          |5           |
|job02         |images_folder |3         |3          |5           |

* `name`: name of the job, to create the checkpoint, the video file, etc.
    * must be unique
* `dataset`: image folders where the input images can be found
    * must be a subfolder of `/data/`
    * all images must be the same size 
    * images are fetched recursively in folders and subfolders
    * you can add multiple folders with a comma: `images_folder1,images_folder2,images_folder3`
* `grid_width`, `grid_height`: the product of these 2 values determines the `batch_size` of the training 
(9 in the above example), produced frames will be output in a grid format (if you use 640x360 images, output frames 
will be 1920x1080 images, in a 3x3 grid format, similar to [this render](https://www.vjloops.com/stock-video/abstract-shutter-3x3-16-137695.html)))
* `video_length`: length of the output video in minutes

A more complete example of CSV config file with can be found [here](example_config.csv).

More columns can be added with the parameters described below:

## Model Parameters

#### Number of Layers
* `nbr_of_layers_g` and `nbr_of_layers_d`: number of layers in the Generator and in the Discriminator.
* Default value: `5`
   
#### Activation Functions
* `activation_g` and `activation_d`: activation functions between layers of the Generator and the Discriminator. 
* Default values: `relu` and `lrelu`.
* Possible values: `relu`, `relu6`, `lrelu`, `elu`, `crelu`, `selu`, `tanh`, `sigmoid`, `softplus`, `softsign`, `softmax`, `swish`.
    
#### Adam Optimizer

* `learning_rate_g`, `beta1_g`, `learning_rate_d`, `beta1_d`: parameters of Adam for the Generator and the Discriminator

|Parameter          |TensorFlow default|DCGAN default|
|---                |---               |---          |
|**Learning rate**  |0.001             |0.0002       |
|**Beta1**          |0.9               |0.5          |

#### Kernel Size

* `k_w` and `k_h`: size of the convolution kernel
* Default value: `5`

#### Batch Normalization

According to the original authors of the model, batch normalization "*deals with poor initialization helps gradient flow*".

* `batch_norm_g` and `batch_norm_d`
* Default value: `True` 

## Video Parameters

* `render_res`: if you use for example 1280x720 images and you picked 2 for `grid_width` and `grid_height`, by default
output frames will be 2560x1440 in a 2x2 grid format. But you can also render 4 videos in 1280x720 by setting 
`render_res` at `1280x720`. The resulting 1280x720 videos are referred as "boxes".
* `auto_render_period`: allow to render videos before the training is completed, so you can preview the result and save 
some disk space (as it quickly produces Gb of images). For example, if you pick `60`, every time it has produces enough 
frames to render 1 minute of video, it will be rendered while the training process continues in parallel.
    * The resulting video files have suffix `_timecut0001.mp4` so they be can merged later
    * You can use the following script `python3 merge_timecuts.py /home/user/Video/folder-with-timecuts` to do that

# Dependencies

## Linux

* `ffmpeg` is required to render the videos
* `mencoder` is only needed to the `merge_timecuts.py` script

## Python libs

This is my current environment, for reference. 
Not all libraries listed here are required, but you'll need at least `tensorflow`, `Pillow`, `pandas`, `numpy`, 
`opencv-python`, `h5py`.

```
(tensorflow4) benoit@farm:~$ pip3 list
Package              Version 
-------------------- --------
absl-py              0.7.1   
astor                0.7.1   
cffi                 1.12.2  
cloudpickle          0.8.1   
gast                 0.2.2   
grpcio               1.19.0  
h5py                 2.9.0   
horovod              0.16.1  
Keras-Applications   1.0.7   
Keras-Preprocessing  1.0.9   
Markdown             3.1     
mock                 2.0.0   
numpy                1.16.2  
opencv-python        4.0.0.21
pandas               0.24.2  
pbr                  5.1.3   
Pillow               5.4.1   
pip                  19.0.3  
protobuf             3.7.1   
psutil               5.6.1   
pycparser            2.19    
python-dateutil      2.8.0   
pytz                 2018.9  
scipy                1.2.1   
setuptools           39.1.0  
six                  1.12.0  
tensorboard          1.13.1  
tensorflow-estimator 1.13.0  
tensorflow-gpu       1.13.1  
termcolor            1.1.0   
Werkzeug             0.15.1  
wheel                0.33.1 
```
## Set-up

* Linux Mint 19.1 (Ubuntu 18.04)
* TensorFlow 1.13.0
* Cuda 10
* cudnn 7.5.0
* Nvidia driver 410.104 
* GeForce GTX 1070 (5x)

# Future Developments / Road map

You're welcome to contribute if you want to. These are the next features I'm planning to work on:

## Model & Config

* Use a YAML instead of a CSV for jobs config, which would allow more complex configurations,
with different settings at different layers, convolution parameters, activation function parameters, etc.; 
use a more "embedded" structure instead of a "flat" CSV structure.
* Allow different types of convolution (dilated convolution, etc.)
* Allow different kernel sizes at different layers.
* Random configuration generator.
* Different colors models:
    * Instead of normalizing the 3 RGB values `[0-255]` to `[-1,1]`, we could maybe normalize a `[0-16M]` color value 
    to a single `[-1,1]` variable (with maybe more precision), in order to reduce GPU memory footprint.
    * Try out alternative color models like HSL, HSV, RGB + alpha, etc.
* Add other optimizers and loss functions to the job config.

## Jobs queuing

* Assign jobs automatically based on available GPU.

## Usability

* Make `--disable_cache` true by default, as caching np images takes a lot of RAM, for a relatively 
small performance improvement (~10%).
* Set-up a default dataset that could be downloaded automatically, so it would be possible to run 
the code without preparing an images dataset. For example movies posters, which had been used
in this other project [benckx/dnn-movie-posters](https://github.com/benckx/dnn-movie-posters), or 
any other freely available images set.

## Performances

* ~~The samples frames are merged before being persisted to the file system, then cut again later before processing the video.
This could be made more efficient.~~

# Related Projects

* Extract face data from images: [benckx/tensorflow-face-detection](https://github.com/benckx/tensorflow-face-detection)
* Clean up images datasets (crop, filter, resize, etc.): [benckx/iapetus-images](https://github.com/benckx/iapetus-images)

# Credit

See [the original project](https://github.com/carpedm20/DCGAN-tensorflow) 
and Taehoon Kim / [@carpedm20](http://carpedm20.github.io/) for more info.
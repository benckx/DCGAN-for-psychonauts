# About
The idea is to:
* Tinker a bit with the code
* Understand how it works
* Interface with [another project](https://github.com/benckx/dnn-movie-posters)
* Create psychedelic videos

# Changes

* Compatible with TensorFlow 1.5.0
* Added parameters `grid_height` and `grid_width`: the size of the grid of the 'train' and 'test' images (in the folder 'samples')
* Parameters `input_height`, `input_width`, `output_height`, `output_width` are set automatically 
(assuming all images in the data set have the same size)
* Added `sample_rate` parameter: how often it creates a sample image ('1' for every iteration, '2' for every other iteration, etc.)
* ... and many many other parameters (activation function, number of layers, etc)

# Usage

    nohup python3 multitrain.py &

# Credits

See [the original project](https://github.com/carpedm20/DCGAN-tensorflow) 
and Taehoon Kim / [@carpedm20](http://carpedm20.github.io/) for more info

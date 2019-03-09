# About
The idea is to:
* Tinker a bit with the [the original project](https://github.com/carpedm20/DCGAN-tensorflow) 
* Understand how it works
* Interface with [another project](https://github.com/benckx/dnn-movie-posters) about movies posters
* Create psychedelic videos (portfolio can be found [here](https://www.avloops.com/users/benckx) and [here](https://www.vjloops.com/users/20585.html))

# New parameters and fixes

* Compatible with TensorFlow 1.5.0
* `grid_height` and `grid_width`: the size of the grid of the 'train' and 'test' images (in the folder 'samples')
* Parameters `input_height`, `input_width`, `output_height`, `output_width` are set automatically 
(assuming all images in the data set have the same size)
* `sample_rate`: how often it creates a sample image ('1' for every iteration, '2' for every other iteration, etc.)
* ... and many many other parameters (activation function, number of layers, loss function, etc)

# Usage

    nohup python3 multitrain.py &

# Credits

See [the original project](https://github.com/carpedm20/DCGAN-tensorflow) 
and Taehoon Kim / [@carpedm20](http://carpedm20.github.io/) for more info

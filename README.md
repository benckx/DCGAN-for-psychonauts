# About
The idea is to:
* Tinker a bit with the [the original project](https://github.com/carpedm20/DCGAN-tensorflow) 
* And understand how it works
* Interface with [another project](https://github.com/benckx/dnn-movie-posters) about movies posters
* Create psychedelic videos (portfolio can be found [here](https://www.avloops.com/users/benckx) and [here](https://www.vjloops.com/users/20585.html))

# Usage

    python3 multitrain.py

# New parameters and fixes

* Compatible with TensorFlow 1.5.0
* `grid_height` and `grid_width`: the size of the grid of the 'train' and 'test' images (in the folder 'samples')
* Parameters `input_height`, `input_width`, `output_height`, `output_width` are set automatically 
(assuming all images in the data set have the same size)
* `sample_rate`: how often it creates a sample image ('1' for every iteration, '2' for every other iteration, etc.)
* ... and many many other parameters (activation function, number of layers, loss function, etc)

## Model Parameters

* Parameter with `_g` refer to the Generator
* Parameter with `_d` refer to the Discriminator

#### Layers
*TODO*

#### Activation Functions
*TODO*

#### Adam Optimizer

* `learning_rate_g`, `beta1_g`, `learning_rate_d`, `beta1_d`:

|Parameter          |TensorFlow default|DCGAN default|
|---                |---               |---          |
|**Learning rate**  |0.001             |0.0002       |
|**Beta1**          |0.9               |0.5          |

# Credit

See [the original project](https://github.com/carpedm20/DCGAN-tensorflow) 
and Taehoon Kim / [@carpedm20](http://carpedm20.github.io/) for more info

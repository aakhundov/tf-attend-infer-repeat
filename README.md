Attend, Infer, Repeat
=====================

Implementation of **AIR** framework proposed in **"Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"** [(Eslami et al., 2016)](https://arxiv.org/abs/1603.08575). The work has been done in equal contributions with [Alexander Prams](https://github.com/aprams).

Concrete (Gumbel-Softmax) distribution proposed in [Maddison et al., 2016](https://arxiv.org/abs/1611.01144) and [Jang et al., 2016](https://arxiv.org/abs/1611.00712) is used for sampling "z_pres" random variable in combination with its continuous relaxation. This avoids back-propagation through discrete "z_pres" with NVIL ([Mnih & Gregor, 2014](https://arxiv.org/abs/1402.0030)) used in the original paper. The model is implemented in TensorFlow.

* **multi_mnist.py** needs to be run before training the model for generation of multi-MNIST dataset.
* **training.py** is a runnable script for training the model with default configuration parameters (passed to a constructor of AIRModel class).
* **demo.py** is a live demo of trained model's performance (using saved parameter values from **model** folder) that allows drawing digits in a  Python GUI and reconstructing them in real time.
* **air/air_model.py** contains extensively configurable AIRModel class.
* **air/transformer.py** is borrowed from [TensorFlow models repository](https://github.com/tensorflow/models/tree/master/transformer).

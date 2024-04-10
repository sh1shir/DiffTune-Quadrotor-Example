This is a DiffTune toolset for controller auto-tuning using sensitivity propagation. This toolset is intended to facilitate users' DiffTune applications in two ways. This repository is an example of an implementation of a quadrotor using a geometric controller.
This repository is a continuation of the original DiffTune program designed in MATLAB, however, this leverages the JAX library in Python. The JAX library is a JAX is Autograd and XLA, brought together for high-performance numerical computing, including large-scale machine learning research. JAX provides us with easy auto-differentiation by leveraging its 'grad' function. We also used its 'jit' function to optimize runtime by compiling functions at an earlier time. 
These functions will be very useful for future applications of compiling with various scenarios parallelly using 'vmap' and implementing meta-learning concepts to generate more accurate parameters. 

Instructions for running:
Required packages: jaxlib, jax, scipy, numpy
Adjust the number of iterations you want and run the file. The code may take a while to run on the first attempt, but improves as iterations go on due to the optimizations of the JAX library. Make sure to configure the settings of JAX properly if you are running on a GPU. 

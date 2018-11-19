# TensorBayes
Repository for Master project:

## Development of a software implementing Bayesian association studies for researchers    
Jonathan Klopfenstein

### Working versions:

- Python (NumPy) based: `NumPyBayes_v3.py`
- TensorFlow based: `TensorBayes_v4.2.py`

### Runtimes (on MacBook Air CPU, 2.2GHz i7):

Conditions: number of samples N = 5000, number of covariates M = 10, number of Gibbs sampling iterations = 5000.

- `NumPyBayes_v2.py`: ~ 10s
- `NumPyBayes_v3.py`: ~ 8s
- `TensorBayes_v3.2.py`: ~ 60s
- `TensorBayes_v3.3.py`: ~ 56s
- `TensorBayes_v4.py`: ~ 50s
- `TensorBayes_v4.1.py`: ~ 53s
- `TensorBayes_v4.2.py`: ~ 47s

#### Google collaboratory runtime (in progress):
| Runtime for TensorBayes versions 	| v3.2   	| v3.3 	| v4     	| v4.1 	|
|:-----------------------------------------------------------------:	|--------	|------	|--------	|------	|
| CPU                                                               	| ~ 60s  	|      	| ~ 45s  	|      	|
| GPU - w/o memory growth                                           	| ~ 160s 	|      	| ~ 195s 	|      	|
| TPU                                                               	| ~ 50s  	|      	| ~ 50s  	|      	|

Note: those runtime measurements are to be taken with caution as the benchmarking method used is subject to changes.

### Under active development:

- `TensorBayes_v5.py`

#### About `TensorBayes_v3.3.py` and `NumPyBayes_v3.py`:
Those versions no longer compute whole dataset matrix multiplication that were needed when sampling the mean (`Emu`) and re-updating the residuals (`epsilon`) after a dataset full pass. Therefore, the mean is no longer sampled and the residuals are updated only during a full pass. This decreases the runtimes of `NumPyBayes_v3.py` and `TensorBayes_v3.3.py` by ~ 2s and ~ 4s, respectively, compared to the previous versions.
  
#### About `TensorBayes_v4.1.py`:    

This version implements the TensorFlow dataset API as well as further optimizations:
- Copy avoidance: When consuming NumPy arrays, the data are embedded in one or more `tf.constant` operations. 
> This works well for a small dataset, but wastes memory---because the contents of the array will be copied multiple times---and can run into the 2GB limit for the tf.GraphDef protocol buffer.    
>  -[Importing data guide](https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays) of TensorFlow documentation.

As the conditions runned above represent a relatively small dataset, this implementation is not required and actually increases the runtime by approximately 3 seconds. However, if consuming large arrays, this implementation might be required.

#### About `TensorBayes_v4.2.py`:    

This version implements `tf.control_dependencies` optimizations in order to minimize to number of `sess.run()` calls (i.e leave the graph a minimal amount of time). As the consumed dataset do not exceed the 2GB limit, copy avoidance is not implemented. The `tf.control_dependencies` implementation reduces the runtime by ~ 3s.

#### About `TensorBayes_v5.py`:   
This version will implements all the optimizations as in `TensorBayes_v4.2.py`, but will stream and read data from disk.

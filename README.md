# TensorBayes
Repositery for Master project:

## Development of a software implementing Bayesian association studies for researchers    
Jonathan Klopfenstein

### Working versions:

- Python (Numpy) based: `NumpyBayes_v3.py`
- TensorFlow based: `TensorBayes_v4.py`

#### About `TensorBayes_v4.py`:    
This version is able to compute normally and retrieve the parameters as `NumpyBayes_v3.py` does,
and implements the TensorFlow dataset API. Compared to the previous version (`TensorBayes_v3.3.py`), runtime has improved by 10 seconds.

### Runtimes (on MacBook Air CPU, 2.2GHz i7):

Conditions: Number of samples N = 5000, number of covariates M = 10, number of Gibbs sampling iterations = 5000.

- `NumpyBayes_v2.py`: ~ 10s
- `NumpyBayes_v3.py`: ~ 8s
- `TensorBayes_v3.3.py`: ~ 60s
- `TensorBayes_v4.py`: ~ 50s

#### Google collaboratory runtimes:
| Runtimes for TensorBayes versions 	| v3.2   	| v3.3 	| v4     	| v4.1 	|
|:-----------------------------------------------------------------:	|--------	|------	|--------	|------	|
| CPU                                                               	| ~ 60s  	|      	| ~ 45s  	|      	|
| GPU - w/o memory growth                                           	| ~ 160s 	|      	| ~ 195s 	|      	|
| TPU                                                               	| ~ 50s  	|      	| ~ 50s  	|      	|

Note: those runtime measurements are to be taken with caution as the benchmarking approach used is subject to changes.

### Under active development:

- `TensorBayes_v4.1.py`
- `TensorBayes_v4.2.py`
  
#### About `TensorBayes_v4.1.py`:    

This version will implement the tensorflow dataset API as well as further optimizations:
- Control dependecies
- Copy avoidance


#### About `TensorBayes_v4.2.py`:    

This version will implements all the optimizations as in `TensorBayes_v4.1.py`, but will stream and read data from disk.


# TensorBayes
Repositery for Master project:

## Development of a software implementing Bayesian association studies for researchers    
Jonathan Klopfenstein

### Working versions:

- Python (Numpy) based: `NumpyBayes_v3.py`
- TensorFlow based: `TensorBayes_v4.py`

#### About `TensorBayes_v4.py`:    
This version is able to compute normally and retrieve the parameters as `NumpyBayes_v3.py` does,
and implements the TensorFlow dataset API.

### Runtimes:
- `NumpyBayes_v2.py`: ~ 10s
- `TensorBayes_v3.3.py`: ~ 60s
- `TensorBayes_v4.py`: ~ 50s

#### Google collaboratory `TensorBayes_v3.2.py` runtimes:
- CPU: ~ 60s
- GPU (w/o memory growth): ~ 160s
- TPU: ~ 50s


### Under active development:

- `TensorBayes_v4.1.py`
- `TensorBayes_v4.2.py`
  
#### About `TensorBayes_v4.1.py`:    

This version will implement the tensorflow dataset API as well as further optimizations:
- Control dependecies
- Copy avoidance


#### About `TensorBayes_v4.2.py`:    

This version will implements all the optimizations as in `TensorBayes_v4.1.py`, but will stream and read data from disk.


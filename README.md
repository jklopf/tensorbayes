# TensorBayes
Repositery for Master project:

## Development of a software implementing Bayesian association studies for researchers    
Jonathan Klopfenstein

### Working versions:

- Python (Numpy) based: `NumpyBayes_v2.py`
- TensorFlow based: `TensorBayes_v3.2.py`

#### About `TensorBayes_v3.2.py`:    
This version is able to compute normally and retrieve the parameters as `NumpyBayes_v2.py` does.

### Runtimes:
- `NumpyBayes_v2.py`: ~ 10s
- `TensorBayes_v3.2.py`: ~ 60s

#### Google collaboratory `TensorBayes_v3.2.py` runtimes:
- CPU: ~ 60s
- GPU (w/o memory growth): ~ 160s
- TPU: ~ 50s


### Under active development:

- `TensorBayes_v3.3.py`
  
#### About `TensorBayes_v3.3.py`:    

This version will implement the tensorflow dataset API instead of
placeholders to feed data, as placeholders are the [least efficient way to feed data into a TensorFlow program](https://www.tensorflow.org/api_guides/python/reading_data).




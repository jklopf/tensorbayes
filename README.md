# TensorBayes
Repositery for Master project:

## Development of a software implementing Bayesian association studies for researchers    
Jonathan Klopfenstein

### Working versions:

- Python based: `NumpyBayes_v2.py`
- Tensorflow based: `TensorBayes_v3.1.py`

#### About `TensorBayes_v3.1.py`:    
This version is actually the first one able to compute in a linear, normal manner.
This is due to proper graph building and graph running separation. As a reminder,
generally speaking, all calls beginning with 'tf.' should be outside of a session (graph computation elements).

However, this version do not actually retrieve the simulated parameters of 
the dataset (sigma2_b goes to inf where it should be near Var(g)/M).

The code able to retrieve the injected parameters will be on the next version: `TensorBayes_v3.2.py`


### Under active development:

- `TensorBayes_v3_2.py`
  
#### About `TensorBayes_v3_2.py`:    
This version should be able to retrieve the simulated parameters and
store the history of the sampling, as `NumpyBayes_v2.py` does. 

The next version will implement the tensorflow dataset API instead of
placeholders to feed data, and will be called: `TensorBayes_v3.3.py` (doesn't exist yet).




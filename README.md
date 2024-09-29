# About

This repository contains code that corresponds with the paper "Assessing Distance-Based Change Detection in Continual Anomaly Detection".

This paper uses the change point detection setup from [https://github.com/CollinCoil/cpd-distances](https://github.com/CollinCoil/cpd-distances) to determine task transitions in concept-agnostic continual anomaly detection tasks. We used `change_point_detector.py` to create a list of predicted change points for each dataset and metric, then we fed that list into `task-agnostic-anomaly-detection.py` to produce results. However, any change point detection algorithm that outputs a list of predicted change points could be easily substituted. 

Use the following link to access the paper: To be added. 

Some datasets used in the paper can be accessed at [https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios](https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios), but some are also provided in this repository. 

The code used to run the change point detection experiments is in the `cpd_distance_trials.py` file. Experimental results beyond those presented the paper can be found in `CAD_Full_Tables.pdf` and `consolidated_data.csv`. 

# Citation
A recommended citation is forthcoming. 

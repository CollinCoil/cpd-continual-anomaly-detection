# About

This repository contains code that corresponds with the paper "Distance-based change point detection for novelty detection in concept-agnostic continual anomaly detection".

This paper uses the change point detection setup from [https://github.com/CollinCoil/cpd-distances](https://github.com/CollinCoil/cpd-distances) to determine task transitions in concept-agnostic continual anomaly detection tasks. We used `change_point_detector.py` to create a list of predicted change points for each dataset and metric, then we fed that list into `task-agnostic-anomaly-detection.py` to produce results. However, any change point detection algorithm that outputs a list of predicted change points could be easily integrated into our workflow. 

All datasets used in the paper can be accessed at [[https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios](https://github.com/lifelonglab/lifelong-anomaly-detection-scenarios)](https://www.kaggle.com/datasets/nyderx/lifelong-continual-learning-for-anomaly-detection), but some are also provided in this repository. 

The code used to run the change point detection experiments is in the `task-agnostic-anomaly-detection.py` file. Experimental results beyond those presented the paper can be found in `CAD_Full_Tables.pdf` and `consolidated_data.csv`. Additionally, we provide heatmaps of lifelong ROC-AUC, backward transfer, and forward transfer for every experiment. 

# Citation
A recommended citation and link to paper are forthcoming. 

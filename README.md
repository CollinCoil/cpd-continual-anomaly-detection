# About

This repository contains code that corresponds with the paper "Distance-based change point detection for novelty detection in concept-agnostic continual anomaly detection".

This paper uses the change point detection setup from [https://github.com/CollinCoil/cpd-distances](https://github.com/CollinCoil/cpd-distances) to determine task transitions in concept-agnostic continual anomaly detection tasks. We used `change_point_detector.py` to create a list of predicted change points for each dataset and metric, then we fed that list into `task-agnostic-anomaly-detection.py` to produce results. However, any change point detection algorithm that outputs a list of predicted change points could be easily integrated into our workflow. 

Most datasets used in the paper can be accessed at [https://www.kaggle.com/datasets/nyderx/lifelong-continual-learning-for-anomaly-detection](https://www.kaggle.com/datasets/nyderx/lifelong-continual-learning-for-anomaly-detection), but some are also provided in this repository. 

The code used to run the change point detection experiments is in the `task-agnostic-anomaly-detection.py` file. Experimental results beyond those presented the paper can be found in `CAD_Full_Tables.pdf` and `consolidated_data.csv`. Additionally, we provide heatmaps of lifelong ROC-AUC, backward transfer, and forward transfer for every experiment. These files demonstrate our pipeline's performance on continual anomaly detection (CAD) task. Additionally, we provide results for our pipeline's performance on the change point detection (CPD) task in the "CPD Perforamnce" directory. Results for the CICIDS dataset are new to this work, and results for the other four datasets are pulled directly from [https://github.com/CollinCoil/cpd-distances](https://github.com/CollinCoil/cpd-distances). 

# Citation
A recommended citation is 
```
@article{coil2025distance,
  title={Distance-based change point detection for novelty detection in concept-agnostic continual anomaly detection},
  author={Coil, Collin and Faber, Kamil and Sniezynski, Bartlomiej and Corizzo, Roberto},
  journal={Journal of Intelligent Information Systems},
  pages={1--39},
  year={2025},
  publisher={Springer}
}
```

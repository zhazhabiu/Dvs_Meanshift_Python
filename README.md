# dvs_meanshift_python
This is an unofficial implementation of 2018 IROS《Real-time clustering and multi-target tracking using event-based sensors》using Python.


Official implementation: https://github.com/fbarranco/dvs_clustering_tracking.git


It's a clustering and tracking using Kalman filters for DVS data (event based camera).


## Prerequisites
python >= 3.0  
numpy  
pandas  
openCV  

## Running dvs_meanshift
```Python
python src/main.py  
```

## Data Preparation
Test data should be put in *'./dataset/{name_of_dataset}/events.txt'*, or you can change the file reading path in *src/main.py*.


## Tracking results
All the trajectories will be saved in *'./{name_of_dataset}_tracking_res'*.
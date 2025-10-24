# RBMTT
Source Code for "Residual-Based Learning Efficient Transformer Network for Maneuvering Target Tracking". If you find this repository helpful in your research, please cite this paper.

* run loaddata.py to load the train data. The data is a three-dimensional matrix, where the first dimension represents the track index, the second dimension represents the time sampling points, and the third dimension represents the measurements or target states. Please generate the data using MATLAB and save it as a .mat file. 
* run RBMTT_train.py to train.
* run RBMTT_infer.py to test. Prepare the test data the same way as the train data.
* The results will be saved in track.mat.

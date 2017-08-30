# Code-for-Parthasarathy-et-al
This repository contains the code that generates the results for cross-temporal decoding, the projections on the PCA space and the identification of mixed selective neurons. All the codes are written in MATLAB.

Decode_final_temporal_v2.m is the code that performs the cross-temporal decoding on a neural population data stored as spike counts. The results from Fig 2a, 2e, 4a, 4f, 6a, 6b, 6c, 6e, 6f, 6g can be generated using this code.

NeuralTraj_v2.m is the code that generates the projections seen in Fig 3 and Fig 6. The result from this code can be used to measure the shift in cluster centers to quanitfy the code morphing.

TwoWayANOVA.m identifies the NMS,LMS and CS neurons from a population of neurons (Fig 5). The neurons identified by this code is further used in Fig 6 for cross-temporal decoding and the analysis on PCA space to measure code morphing.

Please note that these codes serve as guideline if one wants to perform these analyses on their dataset. Please read through the comments in the code to format your dataset to suit the code.  

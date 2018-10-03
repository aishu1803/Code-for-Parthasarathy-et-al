# Code-for-Parthasarathy-et-al
This repository contains the code that generates the results for Parthasarathy, A., Herikstad, R., Bong, J. H., Medina, F. S., Libedinsky, C., & Yen, S.-C. (2017). Mixed selectivity morphs population codes in prefrontal cortex. Nature Publishing Group, 20(12), 1770–1779. http://doi.org/10.1038/s41593-017-0003-2.
All the codes are written in MATLAB.

Decode_final_temporal_v2.m is the code that performs the cross-temporal decoding on a neural population data stored as spike counts. The results from Fig 2a, 2e, 4a, 4f, 6a, 6b, 6c, 6e, 6f, 6g can be generated using this code.

NeuralTraj_v2.m is the code that generates the projections seen in Fig 3 and Fig 6. The result from this code can be used to measure the shift in cluster centers to quanitfy the code morphing.

TwoWayANOVA.m identifies the NMS,LMS and CS neurons from a population of neurons (Fig 5). The neurons identified by this code is further used in Fig 6 for cross-temporal decoding and the analysis on PCA space to measure code morphing.

In addition to these .m files, the repository also contains a .m file called AssignTrialLabel that is an auxilliary function necessary for all these main functions. The comments inside the main function describe the need for AssignTrialLabel.m

Please note that these codes serve as guideline if one wants to perform these analyses on their dataset. Please read through the comments in the code to format your dataset to suit the code.  

Please contact Aishwarya Parthasarathy at aishu.parth@gmail.com if you have any questions.

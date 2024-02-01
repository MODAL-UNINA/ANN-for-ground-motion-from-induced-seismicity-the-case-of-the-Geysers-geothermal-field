# A data-driven artificial neural network model for the prediction of ground motion from induced seismicity: The case of The Geysers geothermal field

Source code for the publication ''A data-driven artificial neural network model for the prediction of ground motion from induced seismicity: the case of the Geysers geothermal field''. DOI: https://doi.org/10.3389/feart.2022.917608.


## Abstract
Ground-motion models have gained foremost attention during recent years for being capable of predicting ground-motion intensity levels for future seismic scenarios. They are a key element for estimating seismic hazard and always demand timely refinement in order to improve the reliability of seismic hazard maps. In the present study, we propose a ground motion prediction model for induced earthquakes recorded in The Geysers geothermal area. We use a fully connected data-driven artificial neural network (ANN) model to fit ground motion parameters. Especially, we used data from 212 earthquakes recorded at 29 stations of the Berkeley-Geysers network between September 2009 and November 2010. The magnitude range is 1.3 and 3.3 moment magnitude (Mw), whereas the hypocentral distance range is between 0.5 and 20 km. The ground motions are predicted in terms of peak ground acceleration (PGA), peak ground velocity (PGV), and 5% damped spectral acceleration (SA) at T=0.2, 0.5, and 1 s. The predicted values from our deep learning model are compared with observed data and the predictions made by empirical ground motion prediction equations developed by Sharma et al. (2013) for the same data set by using the nonlinear mixed-effect (NLME) regression technique. For validation of the approach, we compared the models on a separate data made of 25 earthquakes in the same region, with magnitudes ranging between 1.0 and 3.1 and hypocentral distances ranging between 1.2 and 15.5 km, with the ANN model providing a 3% improvement compared to the baseline GMM model. The results obtained in the present study show a moderate improvement in ground motion predictions and unravel modeling features that were not taken into account by the empirical model. The comparison is measured in terms of both the R2 statistic and the total standard deviation, together with inter-event and intra-event components.

![image](https://github.com/MODAL-UNINA/ANN-for-ground-motion-from-induced-seismicity-the-case-of-the-Geysers-geothermal-field/assets/152622661/e8cc7b44-8989-43ff-8bf0-2af4344ae784)

Geographic map of The Geysers geothermal field, California. Black triangles identify the seismic stations. Gray circles indicate the epicentral location of the earthquakes analyzed in the present study. Circle dimension is proportional to the event magnitude. Gray lines correspond to the known quaternary faults. The red square and the red arrow in the inset indicate the location of The Geysers geothermal field.

## Installation
Check the environment.ylm file.

## Supplementary material
The Supplementary Material for this article can be found online at: https://www.frontiersin.org/articles/10.3389/feart.2022.917608/full#supplementary-material. 

## Acknowledgment
This work has been designed and developed under the “PON Ricerca e Innovazione 2014-2020”– Dottorati innovativi con caratterizzazione industriale XXXVI Ciclo, Fondo per lo Sviluppo e la Coesione, code DOT1318347, CUP E63D20002530006.

This study has been also supported by PRIN-2017 MATISSE Project, No. 20177EPPN2, funded by the Italian Ministry of Education and Research.

NS is also thankful to the support provided by CSIR-National Geophysical Research Institute, Hyderabad, India.

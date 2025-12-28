# **Graph self-supervised embedding for multiscale river network selection**
This repository contains the code and data of the paper.

# Description of the study

River network selection is a critical operator in map generalization, directly influencing map quality. Conventional methods rely on human-defined features, which may introduce subjectivity and fail to capture spatial contextual relationships among rivers, thereby undermining the accuracy of river importance assessment. Although graph deep learning has partially addressed these issues, it depends on labeled training data and often yields topologically fragmented outputs. To address these limitations, this study proposes a multiscale river network selection method based on graph self-supervised learning (GSSL). A heterogeneous river network graph is constructed, and a graph attention-based autoencoder (GAT_AE) is employed to learn river embeddings. River importance is quantified using embedding vector norms, while the radical law model is integrated to facilitate multiscale selection. Empirical evaluations demonstrate that the proposed method outperforms existing approaches, achieving an F1 score of 0.86, an AUC of 0.96, a connected component ratio of 1.00, a line correspondence coefficient of 0.92, and a morphological similarity of 0.97. Overall, our method ensures topological connectivity through a heterogeneous graph model, introduces a data-driven metric for assessing river significance using GAT_AE, and establishes a multiscale selection framework via GSSL without labeled data. It offers a robust and objective solution for intelligent map generalization.

## Data and codes availability statement
Due to the large size of the data, we have provided an alternative download link for the data and code: https://figshare.com/s/32116424c07a82c65e94.

## Description of data
The data folder contains three main components: training data (`0-training data`), test data (`1-test data`), and evaluation results (`2-evaluation result`). A detailed description of the contents in each subfolder is provided below.
### 0-training data
* `0-river network` contains 771 sets of original river network data for training.
* `1-watershed` contains 771 watershed data of the river networks from the USGS National Hydrography Dataset.
* `2-simplified river network` contains the river network data after curve simplification, used for the construction of river strokes.
* `3-adj_csv` contains the adjacent matrices of the heterogeneous river network graphs constructed from the 771 river networks.
* `4-triangle` contains the Delaunay triangulation data generated during the calculation of watershed areas for each reach across the 771 river networks.
* `5-watershed skeleton` contains the linear watershed data for each reach across the 771 river networks.
* `6-watershed for each reach` contains the reach-level watershed data for the 771 river networks.
* `7-river network` contains the 771 river networks recording stroke code, river features, and the results of multiscale river network selection by various methods.
### 1-test data
* `0-river network` contains 9 sets of large river networks for reasoning.
* `1-watershed` contains 9 watershed data of the river networks from the USGS National Hydrography Dataset.
* `2-simplified river network` contains the test river networks after curve simplification, used for the construction of river strokes.
* `3-adj_csv` contains the adjacent matrices of the heterogeneous river network graphs constructed from the 9 test river networks.
* `4-triangle` contains the Delaunay triangulation data generated when calculating the watershed area for each reach of the 9 test river networks.
* `5-watershed skeleton` contains the linear watershed data for each reach within the 9 test river networks.
* `6-watershed for each reach` contains the reach-level watershed data of the 9 test river networks.
* `7-river network` contains the 9 test river networks recording stroke code, river features, and the results of multiscale river network selection by various methods.
### 2-evaluation result
* `0-ML metrics.csv` contains machine-learning metrics evaluating the multiscale river network selection results of GAT_AE, feature ranking methods, machine learning methods, and graph deep learning methods on the 9 test data.
* `1-CLC results.csv` contains the coefficient of line correspondence (CLC) evaluating the multiscale river network selection results of various methods on the 9 test data.
* `2-CCR results.csv` contains connected component ratio (CCR) evaluating the multiscale river network selection results of various methods on the 9 test data.
* `3-similarity_results.csv` contains morphological similarity (SIM) evaluating the multiscale river network selection results of various methods on the 9 test data.
* `9-GAT_AE results.xlsx` contains all evaluation metrics for GAT_AE method on the 9 test data.
* `merged results.xlsx` contains all evaluation metrics for various methods on the 9 test data.

## Description of code
This section outlines the code used in the study, which requires execution across different environments: torch-geometric, ArcGIS, and QGIS. The following describes the core scripts that must be executed independently; auxiliary code components serving as background dependencies are not detailed here.

-------------------------------***torch-geometric Requirement***-------------------------------

- Python3.7
- Numpy
- sklearn
- torch
- torch_geometric
- geopandas
- pandas
- shap
- scipy
- matplotlib
- networkx

--------------------------------------***ArcGIS Requirement***---------------------------------------

- ArcGIS10.6
- Python2.7
- arcpy
- Numpy

---------------------------------------***QGIS Requirement***----------------------------------------

- QGIS3.16
- Python3.7
- Numpy
- pandas
- triangle
- PyQt5
- collections

 
### Building heterogeneous river network graph
* **QGIS/build_heterogenous_graph.py** constructs river strokes and subsequently builds a heterogeneous river network graph based on these strokes.
<br>[Input]：`0-river network` and `2-simplified river network`
<br>[Output]：`3-adj_csv`. The attribute fields `id_stroke` and `toStroke` are added to the river network data in `0-river network`.

* **QGIS/hydroOrdering.py** constructs the Horton code based on the river strokes.
<br>[Input]：`0-river network` and `2-simplified river network`
<br>[Output]：The attribute field `Horton` is added to the river network data in `0-river network`.

* **QGIS/delaunay_skeleton_used.py** constructs the boundary for the reach-level watershed.
<br>[Input]：`0-river network` and `1-watershed`
<br>[Output]：`4-triangle` and `5-watershed skeleton`

* **ArcGIS/batch_line_to_polygon.py** converts the reach-level watershed boundary into a polygonal watershed.
<br>[Input]：`5-watershed skeleton`
<br>[Output]：`6-watershed for each reach`

* **ArcGIS/batch_cal_area.py** calculates the watershed area for each reach.
<br>[Input]：`6-watershed for each reach`
<br>[Output]：The attribute field `Area` is added to the watershed data in `6-watershed for each reach`.

* **ArcGIS/batch_spatial_join.py** assigns the watershed area of each reach to the river network data.
<br>[Input]：`0-river network` and `6-watershed for each reach`
<br>[Output]：The attribute field `Area` is added to the river network in `7-river network`.

* **ArcGIS/batch_cal_length.py** calculates the length for each reach.
<br>[Input]：`7-river network`
<br>[Output]：The attribute field `Length` is added to the river network in `7-river network`.

* **ArcGIS/batch_cal_density.py** calculates the river density for each reach.
<br>[Input]：`7-river network`
<br>[Output]：The attribute field `Density` is added to the river network in `7-river network`.

* **torch-geometric/cal_feature.py** processes reach-level features to derive river stroke-level features.
<br>[Input]：`7-river network`
<br>[Output]：The attribute fields `stk_len`, `stk_area`, `stk_dens`, `stk_sinu`, `stk_recAn`, `stk_upar`, and `stk_tri` are added to the river network in `7-river network`. 

***
### torch-geometric/GAT_AE training and inference
* **torch-geometric/transferData.py** converts the heterogeneous river network graphs into the PyTorch data format to support GAT_AE training and inference.
<br>[Input]：`7-river network` and `3-adj_csv`
<br>[Output]：`training_graphs.pt` and `test_graphs.pt`

* **torch-geometric/train_GAE.py** trains the GAT_AE model.
<br>[Input]：`./code/torch-geometric/GAT_AE/0-torch data/training_graphs.pt`
<br>[Output]：Trained model `best_128_16_3_Adam.pth` and training record `loss_128_16_3_Adam.csv`

* **torch-geometric/loadmodel_L2.py** applies the GAT_AE model for inference.
<br>[Input]：`./data/1-test data/7-river network/7-river network` and `./code/torch-geometric/GAT_AE/0-torch data/test_graphs.pt`
<br>[Output]：The attribute field `128_16_3_I`, which records the river importance ranking results, is added to the river network data in `./data/1-test data/7-river network/7-river network`.

* **torch-geometric/multi-scale selected RN by sort.py** performs multiscale river network selection based on the river importance ranking provided by GAT_AE, integrating the radical law model.
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The attribute field `pre_I`, containing the multiscale selection results, is added to the river network data in `./data/1-test data/7-river network/7-river network`.

***
### Building comparison methods: feature ranking, machine learning, and graph deep learning
* **torch-geometric/multiScale RN selection by feature.py** performs multiscale river network selection using feature ranking methods based on river length, upstream cumulative area, and the number of tributaries.
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The multiscale river network selection results are stored in the attribute fields `S_stk_len`, `S_stk_upar`, and `S_stk_tri` of the test river networks in `./data/1-test data/7-river network/7-river network`.

* **torch-geometric/ML_method.py** performs multiscale river network selection using machine learning methods, including SVM, Tree, and MLP.
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The multiscale river network selection results are recorded in the attribute fields `pre_SVM`, `pre_TREE`, and `pre_MLP` of the test river networks in `./data/1-test data/7-river network/7-river network`.

* **torch-geometric/distance_proximity_river.py** calculates the proximity distance for the 9 test river networks, used for the graph deep learning methods.
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The attribute field `dpr` is added to the test river networks in `./data/1-test data/7-river network/7-river network`.

* **torch-geometric/upstream_area.py** calculates the reach-level upstream cumulative area for the 9 test river networks, used for the graph deep learning methods.
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The attribute field `upArea` is added to the test river networks in `./data/1-test data/7-river network/7-river network`.

* **torch-geometric/train_based_on_reach.py** performs multiscale river network selection using graph deep learning methods, including 1st-chebnet, SAGE, and GAT.
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The multiscale river network selection results `GCNpre`, `SAGEpre`, and `GATpre` are added to the river network data in `./data/1-test data/7-river network/7-river network`.
***
### Calculating evaluation metrics
* **torch-geometric/torch-geometric/ML metrics.py** calculates the machine-learning evaluation metrics for GAT_AE, feature ranking methods, machine learning methods, and graph deep learning methods.
<br>[Input]： `./data/1-test data/7-river network/7-river network`
<br>[Output]：`0-ML metrics.csv`

* **torch-geometric/CLC_metrics.py** calculates the CLC evaluation metrics for GAT_AE, feature ranking methods, machine learning methods, and graph deep learning methods.
<br>[Input]： `./data/1-test data/7-river network/7-river network`
<br>[Output]：`1-CLC results.csv`

* **torch-geometric/cal_ConnectRatio.py** calculates the CCR evaluation metrics for GAT_AE, feature ranking methods, machine learning methods, and graph deep learning methods.
<br>[Input]： `./data/1-test data/7-river network/7-river network`
<br>[Output]：`2-CCR results.csv`

* **torch-geometric/cal_SIM.py** calculates the SIM evaluation metrics for GAT_AE, feature ranking methods, machine learning methods, and graph deep learning methods.
<br>[Input]： `./data/1-test data/7-river network/7-river network`
<br>[Output]：`3-similarity_results.csv`

* **torch-geometric/draw_embedding4bigRN.py** visualizes the river embeddings calculated by the GAT_AE encoder using t-SNE.
<br>[Input]：`./code/torch-geometric/GAT_AE/0-torch data/test_graphs.pt`
<br>[Output]：Figure 8

* **torch-geometric/draw_metrics_bae_curve.py** plots the GAT_AE evaluation metrics as a histogram and curve graph.
<br>[Input]：`./data/2-evaluation result/merged results.xlsx`
<br>[Output]：Figure 11

* **torch-geometric/draw_different_method_curve.py** plots the evaluation metrics curves for all methods.
<br>[Input]：`./data/2-evaluation result/merged results.xlsx`
<br>[Output]：Figure 16

## Instructions for reproducing figures and tables in the manuscript

* **Instructions for reproducing Figure 7:**
Run `train_GAE.py` to train GAT_AE model and get the loss changes during the training process. The data is saved as `loss_128_16_3_Adam.csv`. Then, use Origin software to plot the loss to get Figure 7.
<br>[Code]：`train_GAE.py`
<br>[Input]：`./code/torch-geometric/GAT_AE/0-torch data/training_graphs.pt`
<br>[Output]：`loss_128_16_3_Adam.csv`
<br>[Postprocess]：Use Origin software to visualize the training and validation losses from `loss_128_16_3_Adam.csv`, generating Figure 7.

* **Instructions for reproducing Figure 8:**
Run `draw_embedding4bigRN.py` to visualize the river embeddings for the test river networks, generating Figure 8.
<br>[Code]：`draw_embedding4bigRN.py`
<br>[Input]：`./code/torch-geometric/GAT_AE/0-torch data/test_graphs.pt`
<br>[Output]：Figure 8

* **Instructions for reproducing Figures 9 and 10:**
Run `multi-scale selected RN by sort.py` to predict the multiscale selection results for the test river networks, generating the attribute field `pre_I`. Then, use ArcGIS and CoreDRAW to produce Figures 9 and 10.
<br>[Code]：`multi-scale selected RN by sort.py`
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The attribute field `pre_I`, containing the multiscale selection results, is added to the river network data in `./data/1-test data/7-river network/7-river network`.
<br>[Postprocess]：Use ArcGIS and CoreDRAW software to visualize the attribute field `pre_I`, generating Figures 9 and 10.

* **Instructions for reproducing Figure 11:**
Run `MLmetrics.py`, `CLC_metrics.py`, `cal_ConnectRatio.py` and `cal_SIM.py` to generate `0-ML metrics.csv`, `1-CLC results.csv`, `2-CCR results.csv` and `3-similarity_results.csv`. Based on these CSV files, integrate them into the GAT_AE evaluation metrics file `merged results.xlsx`. Then, use `draw_metrics_bae_curve.py` to plot Figure 11.
<br>[Code]：`MLmetrics.py`, `CLC_metrics.py`, `cal_ConnectRatio.py` and `cal_SIM.py`
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：`0-ML metrics.csv`, `1-CLC results.csv`, `2-CCR results.csv` and `3-similarity_results.csv`
<br>[Postprocess]：Integrate the aforementioned CSV files into the evaluation metrics file `merged results.xlsx`.
<br>[Code]：`draw_metrics_bae_curve.py`
<br>[Input]：`merged results.xlsx`
<br>[Output]：Figure 11

* **Instructions for reproducing Figures 12, 13, and 14:**
Run `multiScale RN selection by feature.py`, `ML_method.py` and `train_based_on_reach.py` to generate the multiscale river network selection results for the feature ranking, machine learning, and graph deep learning methods. These results are added to the attribute fields `S_stk_len`, `S_stk_upar`, `S_stk_tri`, `pre_SVM`, `pre_TREE`, `pre_MLP`, `GCNpre`, `SAGEpre` and `GATpre` of the test river network in `./data/1-test data/7-river network/7-river network`. Then, run `cal_difference.py` to calculate the difference between the human-selected networks and algorithm-selected networks. Finally, use ArcGIS and CorelDRAW to produce Figures 12, 13, and 14.
<br>[Code]：`multiScale RN selection by feature.py`, `ML_method.py` and `train_based_on_reach.py`
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The attribute fields `S_stk_len`, `S_stk_upar`, `S_stk_tri`, `pre_SVM`, `pre_TREE`, `pre_MLP`, `GCNpre`, `SAGEpre` and `GATpre` in `./data/1-test data/7-river network/7-river network`
<br>[Code]：`cal_difference.py`
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：The attribute fields `X_S_stk_len`, `X_S_stk_upar`, `X_S_stk_tri`, `X_pre_SVM`, `X_pre_TREE`, `X_pre_MLP`, `X_GCNpre`, `X_SAGEpre` and `X_GATpre` in `./data/1-test data/7-river network/7-river network`
<br>[Postprocess]：Use ArcGIS and CorelDRAW to visualize the attribute fields, generating Figures 12, 13, and 14

* **Instructions for reproducing Figure 15:**
Integrate `0-ML metrics.csv`, `1-CLC results.csv`, `2-CCR results.csv` and `3-similarity_results.csv` to form `merged results.xlsx`. Then, run `draw_different_method_curve.py` to plot Figure 15.
<br>[Code]：`MLmetrics.py`, `CLC_metrics.py`, `cal_ConnectRatio.py` and `cal_SIM.py`
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：`0-ML metrics.csv`, `1-CLC results.csv`, `2-CCR results.csv` and `3-similarity_results.csv`
<br>[Postprocess]：Integrate the aforementioned CSV files into the evaluation metrics file `merged results.xlsx`.
<br>[Code]：`draw_different_method_curve.py`
<br>[Input]：`merged results.xlsx`
<br>[Output]：Figure 15

* **Instructions for reproducing Table 2:**
Based on river network data in `data/0-training data/7-river network` and `data/1-test data/7-river network`, use the Field Calculator in ArcGIS to count the number of river strokes (recorded in the attribute field `id_stroke`) and river network length to get Table 2. 

* **Instructions for reproducing Table 3:**
Based on `merged results.xlsx`, calculate the mean of the metrics for the test river networks to get Table 3.
<br>[Code]：`MLmetrics.py`, `CLC_metrics.py`, `cal_ConnectRatio.py` and `cal_SIM.py`
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：`0-ML metrics.csv`, `1-CLC results.csv`, `2-CCR results.csv` and `3-similarity_results.csv`.
<br>[Postprocess]：Integrate the aforementioned CSV files into the evaluation metrics file `merged results.xlsx` and calculate the mean value of each evaluation metric.

* **Instructions for reproducing Table 4:** Run `cal_mean_metrics.py` to calculate the mean value of the evaluation metrics for each method to get Table 4, based on `merged results.xlsx`.
<br>[Code]：`MLmetrics.py`, `CLC_metrics.py`, `cal_ConnectRatio.py` and `cal_SIM.py`
<br>[Input]：`./data/1-test data/7-river network/7-river network`
<br>[Output]：`0-ML metrics.csv`, `1-CLC results.csv`, `2-CCR results.csv` and `3-similarity_results.csv`
<br>[Postprocess]：Integrate the aforementioned CSV files into the evaluation metrics file `merged results.xlsx`.
<br>[Code]：`cal_mean_metrics.py` 
<br>[Input]：`merged results.xlsx`
<br>[Output]：Table 4

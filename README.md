---
title: "A Spatially Structured Empirical Bayes Framework for Traffic Safety Countermeasure Evaluation"
author: "Mingjian Wu"
date: "8/26/2025"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Introduction

All codes used in the presented study (i.e., the Network Process Convolution-based Empirical Bayes Method or EB-NPC to evaluate safety effectiveness of traffic countermeasures) are shared here.

For detailed explanations of the NPC model, please refer to its original work: 
Rezaee, H., Schmidt, A. M., Stipancic, J., & Labbe, A. (2022). A process convolution model for crash count data on a network. Accident Analysis & Prevention, 177, 106823.
https://www.sciencedirect.com/science/article/pii/S0001457522002585
https://github.com/aurelielabbe/Process-convolution-model-for-crash-counts-data-on-a-network

We assume the following are available prior to the implementation:

1. Collision data: with longitude, latitude, and severity (optional)

2. Covariates: for example, traffic volume, lane numbers, etc.

3. Road network shape files: ideally, the road shapefile should split by intersections (arterial & collector) only


## Sample data

All files starting with "0_" are sample data (only partial data of what was used in the case study) due to the privacy policy of the City of Edmonton (CoE).

In the python scripts, these two files are used frequently:  and "0_sp_lookups.pkl":
  "0_paths_with_legs_ALL_combined.jsonl.gz": is the "graph" of the CoE's road network built by the road network shapefile of CoE, python libraries "networkx", "shapely" and "geopandas"
  "0_sp_lookups.pkl": cotains the shortest path and intersection information for each pair of points (DFS sites and nonDFS sites)
However, these files exceed the file size limit, so they are allowed to be uploaded here.

## Implementation

All steps documented in the paper are shared here, the numbers in the filenames indicate the order of the implementation.

Feel free to contact us at mingjian.wu@mcgill.ca should you have any questions or ideas to improve the implementation






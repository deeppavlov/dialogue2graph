# Graph incrementation on data sample

## Experiment setup

The incremental approach was tested with the model o3-mini without specifying the temperature. Test data are the first 5 examples from synthetic data subset called "generated_24". The examples include 6-13 dialogs per graph. Incrementation pipeline is as in the exp2025_03_12_rec_models_incrementation. 

## Results and observations

Incrementation quality is suboptimal. Visualizations of true graph and the last generated graph are clearly not the same, but similarity metrics are good in two cases out of five. 

## Future plans

1. Change the visualization library from kawaii to graphviz for better graph visualization.
2. Repeat an experiment with chatgpt-4o-latest.
3. Test other pipelines:
    - take the longest dialog as the first one and continue as before
    - take all the dialogs at once and generate a general graph of them
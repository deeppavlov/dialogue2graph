# Getting started with dialogue augmentation with full graph coverage

## Goal

Goal is to develop an algorithm for determining the minimum number of augmented dialogues required to fully cover the corresponding graph.

## Results

A basic algorithm for covering a graph with dialogues has been developed. The starting dialogue is the dialogue with the maximum number of lines. Each new dialogue to add is determined by the maximum number of new elements (which are new nodes and edges).

## Future plans

In the future, it is planned to add this algorithm into dev branch as a new class. Also, it may be reasonable to get starting dialogue by maximum number of unique nodes and edges, and not lines.
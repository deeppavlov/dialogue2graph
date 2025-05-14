# Graph coverage + utterances similarities

## Goal

The goal is to develop an algorithm that will select the minimum number of dialogs required to completely cover the graph. Additionally, the most diverse dialogs should be selected from the dialogs with one path.

## Results

Basic steps for an algorithm for covering a graph with the most diverse dialogs have been developed. The starting dialog is the dialog with the maximum number of unique nodes and edges. New path to add is selected by the maximum number of new nodes and edges. New dialog to add is selected by the least similarities between shared elements of the dialogs.

## Future plans

In the future, it is planned to add this algorithm into dev branch as a new class. 
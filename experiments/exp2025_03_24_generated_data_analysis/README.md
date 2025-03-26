# D2G_generated analysis

## Basic statistics

* Mean, std, min, max for number of graph edges and nodes, dialogues used for graph creation, mean dialogue messages

|          |Graph edges|Graph nodes|Dialogue number|Mean dialogue messages|
|----------|-----------|-----------|---------------|----------------------|
|mean      |12.50      |8.91       |11.20          |11.80                 |
|std       |3.84       |1.81       |10.08          |3.33                  |
|min       |7.00       |5.00       |2.00           |6.00                  |
|25%       |10.00      |8.00       |6.00           |9.66                  |
|50%       |12.00      |8.00       |9.00           |11.36                 |
|75%       |14.00      |10.00      |12.00          |13.33                 |
|max       |44.00      |24.00      |96.00          |31.00                 |

* Number of graph edges is usually 1.25-1.4 times greater than number of graph nodes, thus, the graphs tend to have tree structure with few interconnections

* There is one unusual graph with 96 dialogues in it on topic 'Requesting a prorated refund for early cancellation' (idx 214)

* There is one unusual graph with 7 long dialogues (mean dialogue length = 31 mes.) on topic 'Requesting sign language interpreter services.' (idx 313)

## Greeting node/edge in the middle of the graph problem

* In D2G data no greeting node appear in the middle of the graph, same goes for edges.
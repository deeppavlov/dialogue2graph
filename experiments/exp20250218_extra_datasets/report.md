# Check graph from dialogue generation on some existing datasets
experiments/exp20250218_extra_datasets/check_datasets_2.ipynb

## Issues and goals
While generating own dialogue dataset 100% accuracy on graph from dialogues generation has been obtained
So question then: what's the problem with our data and what can be done to improve them
It was decided to generate graphs from upto 10 dialogues

## Hypotheses and steps
- Some of dialogue datasets was selected to try different approaches
- From the first sight it was clear that algorithmic method with comparing immediate context won't work for the reason
of data diversity unike our own dataset
- Dialogues here contain more live details like names, days of week, hours, locations, car makes, etc.
Definitely it is the task for LLM.
So semi-ready prompt was modified.
- Experiments with o1-mini prompt showed that model understands better general direction rather than exact instructions
Showing examples also was not succesfull for prompts
- Graph is generated in 1 step: 1 prompt with all the dialogues together

## Data
10 first dialogues from each dataset under selected theme were used
Though dialogues are of same topic, they could have different scenarios and no common nodes at all

## Results

1. MULTIWOZ2_2:restaurant:
```python
load_dataset('Salesforce/dialogstudio', 'MULTIWOZ2_2')
```
graphs with 1-4 dialogues succesfully generated 
graphs from 5-6 dialogues: wrongly combined different restaurants in one node
graph from 7 dialogues has been generated, but not all common nodes found
graph from 8 dialogues completed with error: lost nodes

2. META_WOZ:BOOKING_FLIGHT
```python
load_dataset('microsoft/meta_woz')
```
graphs from 2-3, 5-6 dialogues generated successfully
4 dialogues graph: wrongly combined node
graph from 7 dialogues completed with error: lost nodes

3. SCHEMA:RentalCars_3:
```python
load_dataset("GEM/schema_guided_dialog")
```
graph from 2 dialogues has been successfully generated
graph from 3 dialogues completed with error: lost nodes

4. TaskMaster3:MovieTickets:
```python
dataset = load_dataset("google-research-datasets/taskmaster3"
```
graphs from 1-3,5 dialogues have been successfully generated
graphs from 7,10 nodes: built with prolem: duplicated nodes 

5. MSR-E2E:taxi
https://github.com/xiul-msr/e2e_dialog_challenge/tree/master/data

6 taxi dialogues graph built with problem: wrongly combined different locations in one node
graph from 7 dialogues completed with error: lost nodes

6. SMD:weather:
https://github.com/IBM/naturalistic-variation-goal-oriented-dialog-datasets/tree/main/kvret_dataset_public_updated

graphs from 5,9 dialogues succesfully generated
graph from 10 dialogues completed with error: lost nodes

7. Frames:
https://www.microsoft.com/en-us/research/project/frames-dataset/download/
graph from 1 dialogue successfully generated
graphs from 3-5 dialogues gnerated: common place - availability status missed
graph from 6 dialogues completed with error: lost nodes

8. WOZ: 10 dialogues generated, wrong common nodes
```python
load_dataset('PolyAI/woz_dialogue', 'en')
```
graph from 10 dialogues generated with error:
node 2 wrongly combined different cuisines

## Future plans

All things to be considered by future researchers, plans on next experiments and so on
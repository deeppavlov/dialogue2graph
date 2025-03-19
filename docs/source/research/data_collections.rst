Datasets
========

We provide several datasets that can be utilized for your experiments.

Task oriented dialogue datasets
-------------------------------

If you would like to generate graphs from a set of dialogues, you may utilize the following datasets reformatted for the convenient use.

- WOZ
- MULTIWOZ2_2
- MetaLWOz
- Microsoft Dialogue Challenge
- schema_guided_dialog
- Stanford Multi-Domain
- TaskMaster3
- Frames

**Format**: ['domain', 'dialogue_id', 'dialogue']

**Link**: https://huggingface.co/datasets/DeepPavlov/d2g_real_dialogues

Usage
.....

Load reformatted MULTIWOZ2_2 dialogue dataset

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("DeepPavlov/d2g_real_dialogues", "MULTIWOZ2_2", token=True)


Synthetic dialogue graph dataset
--------------------------------

We generated a dataset that contains 402 dialogue graphs. 

**Format**: ['graph', 'topic', 'dialogues']

**Link**: https://huggingface.co/datasets/DeepPavlov/d2g_generated

Usage
.....

Load reformatted d2g_generated dataset

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)
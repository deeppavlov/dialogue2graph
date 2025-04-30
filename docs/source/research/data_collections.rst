Datasets
========

We provide several datasets that can be utilized for your experiments.

Task oriented dialog datasets
-------------------------------

If you would like to generate graphs from a set of dialogs, you may utilize the following datasets reformatted for the convenient use.

- WOZ
- MULTIWOZ2_2
- MetaLWOz
- Microsoft Dialog Challenge
- schema_guided_dialog
- Stanford Multi-Domain
- TaskMaster3
- Frames

**Format**: ['domain', 'dialog_id', 'dialog']

**Link**: https://huggingface.co/datasets/DeepPavlov/d2g_real_dialogs

Usage
.....

Load reformatted MULTIWOZ2_2 dialog dataset

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("DeepPavlov/d2g_real_dialogs", "MULTIWOZ2_2", token=True)


Synthetic dialog graph dataset
--------------------------------

We generated a dataset that contains 402 dialog graphs. 

**Format**: ['graph', 'topic', 'dialogs']

**Link**: https://huggingface.co/datasets/DeepPavlov/d2g_generated

Usage
.....

Load reformatted d2g_generated dataset

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)
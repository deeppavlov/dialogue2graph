Datasets
========

We provide several datasets that can be utilized for your experiments.

.. |huggingface| image:: ../_static/images/logo-colab.svg
    :align: middle
    :width: 40

Task oriented dialog datasets
-------------------------------

+---------------+-------------------------------------------------------------------------------------------------------------------+
|d2g_real_dialogs                                                                                                                 |
+===============+===================================================================================================================+
|**Description**| Reformatted task oriented dialog datasets                                                                         |
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Datasets**   | 1. WOZ, 2. MULTIWOZ2_2, 3. MetaLWOz, 4. Microsoft Dialog Challenge, 5. schema_guided_dialog, 6. Stanford        |
|               | Multi-Domain, 7. TaskMaster3, 8. Frames                                                                           |
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Format**     |``['domain', 'dialog_id', 'dialog']``                                                                          |
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Link**       | |huggingface| `d2g_real_dialogs on HuggingFace <https://huggingface.co/datasets/DeepPavlov/d2g_real_dialogs>`_|
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Usage**      |.. code-block:: python                                                                                             |
|               |                                                                                                                   |
|               |    from datasets import load_dataset                                                                              |
|               |                                                                                                                   |
|               |    dataset = load_dataset("DeepPavlov/d2g_real_dialogs",                                                        |
|               |                           "MULTIWOZ2_2", token=True)                                                              |
+---------------+-------------------------------------------------------------------------------------------------------------------+

Synthetic dialog graph dataset
--------------------------------

+---------------+---------------------------------------------------------------------------------------------------------+
|d2g_generated                                                                                                            |
+===============+=========================================================================================================+
|**Description**| LLM generated dialog graphs                                                                             |
+---------------+---------------------------------------------------------------------------------------------------------+
|**Format**     |``['graph', 'topic', 'dialogs']``                                                                      |
+---------------+---------------------------------------------------------------------------------------------------------+
|**Link**       | |huggingface| `d2g_generated on HuggingFace <https://huggingface.co/datasets/DeepPavlov/d2g_generated>`_|
+---------------+---------------------------------------------------------------------------------------------------------+
|**Usage**      |.. code-block:: python                                                                                   |
|               |                                                                                                         |
|               |    from datasets import load_dataset                                                                    |
|               |                                                                                                         |
|               |    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)                                       |
+---------------+---------------------------------------------------------------------------------------------------------+

Datasets
========

We provide several datasets that can be utilized for your experiments.

Task oriented dialog datasets
-------------------------------

+---------------+-------------------------------------------------------------------------------------------------------------------+
|d2g_real_dialogues                                                                                                                 |
+===============+===================================================================================================================+
|**Description**| Reformatted task oriented dialog datasets                                                                         |
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Datasets**   | 1. WOZ, 2. MULTIWOZ2_2, 3. MetaLWOz, 4. Microsoft Dialogue Challenge, 5. schema_guided_dialog, 6. Stanford        |
|               | Multi-Domain, 7. TaskMaster3, 8. Frames                                                                           |
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Format**     |``['domain', 'dialogue_id', 'dialogue']``                                                                          |
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Link**       | |huggingface| `d2g_real_dialogues on HuggingFace <https://huggingface.co/datasets/DeepPavlov/d2g_real_dialogues>`_|
|               |.. |huggingface| image:: ../_static/images/logo-colab.svg                                                          |
|               |    :align: middle                                                                                                 |
|               |    :width: 40                                                                                                     |
+---------------+-------------------------------------------------------------------------------------------------------------------+
|**Usage**      |.. code-block:: python                                                                                             |
|               |                                                                                                                   |
|               |    from datasets import load_dataset                                                                              |
|               |                                                                                                                   |
|               |    dataset = load_dataset("DeepPavlov/d2g_real_dialogues",                                                        |
|               |                           "MULTIWOZ2_2", token=True)                                                              |
+---------------+-------------------------------------------------------------------------------------------------------------------+

Synthetic dialogue graph dataset
--------------------------------

+---------------+---------------------------------------------------------------------------------------------------------+
|d2g_generated                                                                                                            |
+===============+=========================================================================================================+
|**Description**| LLM generated dialog graphs                                                                             |
+---------------+---------------------------------------------------------------------------------------------------------+
|**Format**     |``['graph', 'topic', 'dialogues']``                                                                      |
+---------------+---------------------------------------------------------------------------------------------------------+
|**Link**       | |huggingface| `d2g_generated on HuggingFace <https://huggingface.co/datasets/DeepPavlov/d2g_generated>`_|
|               |.. |huggingface| image:: ../_static/images/logo-colab.svg                                                |
|               |    :align: middle                                                                                       |
|               |    :width: 40                                                                                           |
+---------------+---------------------------------------------------------------------------------------------------------+
|**Usage**      |.. code-block:: python                                                                                   |
|               |                                                                                                         |
|               |    from datasets import load_dataset                                                                    |
|               |                                                                                                         |
|               |    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)                                       |
+---------------+---------------------------------------------------------------------------------------------------------+

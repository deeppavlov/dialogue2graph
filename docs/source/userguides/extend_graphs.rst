:tutorial_name: dialog_extension/dialog_extender_example.ipynb

Extend graphs
================

If you would like to extend existing dialog graph, :py:class:`~dialog2graph.pipelines.d2g_extender.LLMGraphExtender` is a class to use.
This class aims to invoke LLMs for increasing node and edge number in a dialog graph.

First, load your data and parse it with :py:class:`~dialog2graph.pipelines.helpers.parse_data.RawDGParser`.

.. code-block:: python

    from datasets import load_dataset
    from dialog2graph.pipelines.helpers.parse_data import RawDGParser, PipelineRawDataType

    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)
    data_example = dataset['train'][0]
    data_example = PipelineRawDataType(dialogs=data_example['dialogs'], supported_graph=data_example['graph'])

    parser = RawDGParser()
    parsed_data = parser.invoke(data_example)

Then, you can pass the data to :py:class:`~dialog2graph.pipelines.d2g_extender.LLMGraphExtender` and get extended graph. Set ``enable_evals`` to True to 
get metric report (it may slow down response time).

.. code-block:: python

    from dialog2graph.pipelines.model_storage import ModelStorage
    from dialog2graph.pipelines.d2g_extender.three_stages_extender import LLMGraphExtender

    ms = ModelStorage()
    graph_extender = LLMGraphExtender(model_storage=ms)
    
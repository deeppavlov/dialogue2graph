Getting started
===============

Installation
~~~~~~~~~~~~

To install the **Dialog2Graph** project, please follow the steps below:

- Install poetry v. 1.8.4 (`detailed installation guide <https://python-poetry.org/docs/>`_) 

.. code-block:: bash

    pipx install poetry==1.8.4

- Clone this repo and install project dependencies

.. code-block:: bash

    git https://github.com/deeppavlov/dialog2graph.git
    cd dialog2graph
    poetry install

- Consider installing `PyGraphviz` from `here <https://pygraphviz.github.io/>`_, if you are planning to visualize your graphs. Then add PyGraphviz to the poetry environment.

.. code-block:: bash

    poetry add pygraphviz

- Ensure that dependencies were installed correctly by running any Python script

.. code-block:: bash

    poetry run python <your_file_name>.py

- Create ``.env`` file to store credentials

.. note::

    Never hardcode your personal tokens and other sensitive credentials. Use the ``.env`` file to store them.

- Explore :doc:`user guides page <userguides>` to start your work.

Key Concepts
~~~~~~~~~~~~

A **dialog graph** is a structured, graph-based representation of conversational data, where **nodes** represent key elements such 
as utterances, intents, or states, and **edges** define the transitions between these elements.

Quickstart
===========

Installation
------------

- Install poetry v. 1.8.4 (`detailed installation guide <https://python-poetry.org/docs/>`_) 

.. code-block:: bash

    pipx install poetry==1.8.4

- Clone this repo and install project dependencies

.. code-block:: bash

    git clone https://github.com/deeppavlov/chatsky-llm-autoconfig.git
    cd chatsky-llm-autoconfig
    poetry install

- Ensure that dependencies were installed correctly by running any Python script

.. code-block:: bash

    poetry run python <your_file_name>.py

- Create ``.env`` file to store credentials

.. note::

    Never hardcode your personal tokens and other sensitive credentials. Use the ``.env`` file to store them.

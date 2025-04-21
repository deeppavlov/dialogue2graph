# How to contribute experiments

Once you've decided to create an experiment to test a hypothesis you should stick to these steps:

1. Create a branch named as `exp/<your hypothesis short summary>`. For example `exp/concatenating_subchains_prompt`
2. After checkout to this newly created branch move to `./experiments` folder
3. Run the following with your experiment name as `exp<YYYY>_<MM>_<DD>_<hypothesis>`

    ```bash
    poetry new <experiment_name>
    ```

    For example:

    ```bash
    poetry new exp2025_01_01_concatenating_subchains_prompt
    ```

4. Now you can go into your experiment folder and initialize your virtual environment

    ```bash
    cd ./exp2025_01_01_concatenating_subchains_prompt
    poetry add ../../
    poetry add <your_dependencies>
    poetry install
    ```

5. Add directories for data and metrics to your folder so it should look like so:

    ```bash
    experiment_name/
    ├── data/
    ├── metrics/
    ├── experiment_name/
        └── your_notebook.ipynb
    ├── README.md
    └── pyproject.toml
    ```

6. You can work in the Jupyter notebooks, but if needed you can create a pipeline folder from the `dialogue2graph/pipelines`. It should resemble this kind of structure if you want to bring results of your experiments as a new pipeline for the library:

    ```bash
    pipeline_name/
    ├── dialogue.py
    ├── graph.py
    ├── pipeline.py
    └── prompts
        └── prompts.py
    ```

7. Don't forget to track your progress and results in the README.md file in your experiment folder.
8. Once you've finished your work create a pull request into `dev` branch. Don't forget to describe what you have done in the experiment in the PR description.

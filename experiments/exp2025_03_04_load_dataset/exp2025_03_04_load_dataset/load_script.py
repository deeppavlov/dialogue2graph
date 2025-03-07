# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script for loading real datasets for d2g task"""


import os
import pandas as pd
import json
from pathlib import Path
import datasets
import urllib.request
import zipfile
import ast

from settings import EnvSettings
env_settings = EnvSettings()
datasets.disable_caching()

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {Dialogue dataset for dialogue2graph task},
author={DeepPavlov.
},
year={2025}
}
"""

_DESCRIPTION = """\
This dataset is composed of these configurations:

"META_WOZ": microsoft/meta_woz,
"SCHEMA": GEM/schema_guided_dialog,
"SMD": https://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip,
MULTIWOZ2_2: Salesforce/dialogstudio:MULTIWOZ2_2,
TaskMaster3: google-research-datasets/taskmaster3,
Frames: https://download.microsoft.com/download/4/b/f/4bf15895-4152-4008-8e68-605912cf7a65/Frames-dataset.zip,
MSR-E2E: https://github.com/xiul-msr/e2e_dialog_challenge/tree/master/data,
WOZ: PolyAI/woz_dialogue:en.
This dataset is designed to develop dialogue 2 graph solution.
"""

_HOMEPAGE = "https://huggingface.co/datasets/DeepPavlov/d2g_real_dialogues"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.



def split2json(split: pd.DataFrame, domain: str) -> list[dict]:
    """ Processes pandas dataframe into JSON with dialogues in necessary format
    """

    new_data = []
    session = split['session.ID'][0]
    cur =  [{'text':"Hello! How can I help you?", "participant":"assistant"}]
    for idx, row in split.iterrows():
       if row["session.ID"] != session:
           new_data.append({"domain": domain, "dialogue": cur})
           cur =  [{'text':"Hello! How can I help you?", "participant":"assistant"}]
       if row["Message.From"] == 'agent':
           cur.append({"text":row['Message.Text'],"participant":"assistant"})
       else:
           cur.append({"text":row['Message.Text'],"participant":row["Message.From"]})
       session = row["session.ID"]
    return new_data


def take_e2e(path_to_files):
    """ Load MSR-E2E .tsv files from path_to_files, handles and saves dataset there
    """
    files = [p for p in Path(path_to_files).iterdir() if p.is_file() and str(p).endswith(".tsv")]
    json_data = []
    for file in files:
        domain = str(file).split("/")[-1].split("_")[0]
        data = pd.read_csv(file, delimiter="\t", usecols=[0,1,2,3,4])
        data = split2json(data, domain)
        json_data.extend(data)
    with open(Path(path_to_files).joinpath("all.json"), 'w') as f:
        f.write(json.dumps(json_data))   
    dataset = datasets.load_dataset("json", data_files=path_to_files+"/all.json")
    train_test = dataset["train"].train_test_split(test_size=0.1)
    valid_test = train_test['test'].train_test_split(test_size=0.5)
    dataset['train'].to_json(Path(path_to_files).joinpath("train.jsonl"))
    valid_test['train'].to_json(Path(path_to_files).joinpath("dev.jsonl"))
    valid_test['test'].to_json(Path(path_to_files).joinpath("test.jsonl"))

def load_frames(path_to_file):

    with open(path_to_file) as f:
        json_data = json.load(f)
    data = []
    for d in [el['turns'] for el in json_data]:
        exist = {"dialogue":
                 [{'text':"Hello! How can I help you?", "participant":"assistant"}] + [{"text":u['text'],"participant":"user"}
                                                                                 if u['author']=="user" else {"text":u['text'], "participant":"assistant"}
                                                                                 for u in d]}
        data.append(exist)
    return data

class d2gRealDialogues(datasets.GeneratorBasedBuilder):
    """Dataset compiled to test d2g task"""

    VERSION = datasets.Version("0.1.0")
    COUNTER = {}


    BUILDER_CONFIGS = [
        datasets.BuilderConfig("SMD", version=VERSION, description="This part of my dataset covers Stanford Multi-Domain Dataset"),
        datasets.BuilderConfig("MULTIWOZ2_2", version=VERSION, description="This part of my dataset covers MultiWOZ 2.2"),
        datasets.BuilderConfig("META_WOZ", version=VERSION, description="This part of my dataset covers MetaLWOz"),
        datasets.BuilderConfig("SCHEMA", version=VERSION, description="This part of my dataset covers The Schema-Guided Dialog dataset"),
        datasets.BuilderConfig("MSR-E2E", version=VERSION, description="This part of my dataset covers Microsoft Dialogue Challenge: Building End-to-End Task-Completion Dialogue Systems"),
        datasets.BuilderConfig("Frames", version=VERSION, description="This part of my dataset covers https://www.microsoft.com/en-us/research/project/frames-dataset/"),
        datasets.BuilderConfig("TaskMaster3", version=VERSION, description="This part of my dataset covers taskmaster3 movie ticketing dialogs"),
        datasets.BuilderConfig("WOZ", version=VERSION, description="This part of my dataset covers WOZ dialogue state tracking dataset"),
    ]

    DEFAULT_CONFIG_NAME = "SMD"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        datasets.disable_caching()
        features = datasets.Features(
            {
                "domain": [datasets.Value("string")],
                "dialogue_id": datasets.Value("string"),
                "dialogue": [{"text": datasets.Value("string"), "participant": datasets.Value("string")}]
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _load_path(self, save_path: str) -> None:
        """Loads dataset into local save_path directory. If save_path exists, does nothing.
        In case of MSR-E2E, just reads .tsv files from save_path, handles and saves dataset there.
        """

        my_dir = Path(save_path)
        if not my_dir.is_dir():
            my_dir.mkdir(parents=True, exist_ok=True)
            if self.config.name == "SMD":
                urllib.request.urlretrieve("https://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip",
                                           my_dir.joinpath("kvret_dataset_public.zip"))
                with zipfile.ZipFile(my_dir.joinpath("kvret_dataset_public.zip"), 'r') as zip_ref:
                    zip_ref.extractall(save_path)
                dataset = datasets.load_dataset("json", data_dir=save_path)
                dataset['train'].to_json(my_dir.joinpath("train.jsonl"))
                dataset['validation'].to_json(my_dir.joinpath("dev.jsonl"))
                dataset['test'].to_json(my_dir.joinpath("test.jsonl"))
            elif self.config.name == "MULTIWOZ2_2":
                dataset = datasets.load_dataset('Salesforce/dialogstudio', 'MULTIWOZ2_2',
                                                token=env_settings.HUGGINGFACE_TOKEN, trust_remote_code=True)
                dataset['train'].to_json(my_dir.joinpath("train.jsonl"))
                dataset['validation'].to_json(my_dir.joinpath("dev.jsonl"))
                dataset['test'].to_json(my_dir.joinpath("test.jsonl"))
            elif self.config.name == "META_WOZ":
                dataset = datasets.load_dataset('microsoft/meta_woz', trust_remote_code=True)
                valid_test_dataset = dataset['test'].train_test_split(test_size=0.5)
                dataset['train'].to_json(my_dir.joinpath("train.jsonl"))
                valid_test_dataset['train'].to_json(my_dir.joinpath("dev.jsonl"))
                valid_test_dataset['test'].to_json(my_dir.joinpath("test.jsonl"))
            elif self.config.name == "SCHEMA":
                dataset = datasets.load_dataset("GEM/schema_guided_dialog", trust_remote_code=True)
                dataset['train'].to_json(my_dir.joinpath("train.jsonl"))
                dataset['validation'].to_json(my_dir.joinpath("dev.jsonl"))
                dataset['test'].to_json(my_dir.joinpath("test.jsonl"))
            elif self.config.name == "TaskMaster3":
                dataset = datasets.load_dataset("google-research-datasets/taskmaster3", trust_remote_code=True)
                test_train_dataset = dataset['train'].train_test_split(test_size=0.1)
                valid_test_dataset = test_train_dataset['test'].train_test_split(test_size=0.5)
                test_train_dataset['train'].to_json(my_dir.joinpath("train.jsonl"))
                valid_test_dataset['train'].to_json(my_dir.joinpath("dev.jsonl"))
                valid_test_dataset['test'].to_json(my_dir.joinpath("test.jsonl"))
            elif self.config.name == "Frames":
                urllib.request.urlretrieve(
                    "https://download.microsoft.com/download/4/b/f/4bf15895-4152-4008-8e68-605912cf7a65/Frames-dataset.zip",
                                           my_dir.joinpath("Frames-dataset.zip")
                )
                with zipfile.ZipFile(my_dir.joinpath("Frames-dataset.zip"), 'r') as zip_ref:
                    zip_ref.extractall(save_path)
                json_data = load_frames(my_dir.joinpath("Frames-dataset").joinpath("frames.json"))
                dataset = datasets.Dataset.from_list(json_data)

                test_train_dataset = dataset.train_test_split(test_size=0.1)
                valid_test_dataset = test_train_dataset['test'].train_test_split(test_size=0.5)
                test_train_dataset['train'].to_json(my_dir.joinpath("train.jsonl"))
                valid_test_dataset['train'].to_json(my_dir.joinpath("dev.jsonl"))
                valid_test_dataset['test'].to_json(my_dir.joinpath("test.jsonl"))
            elif self.config.name == "WOZ":
                dataset = datasets.load_dataset('PolyAI/woz_dialogue', 'en')
                print(dataset)
                dataset['train'].to_json(my_dir.joinpath("train.jsonl"))
                dataset['validation'].to_json(my_dir.joinpath("dev.jsonl"))
                dataset['test'].to_json(my_dir.joinpath("test.jsonl"))

        elif self.config.name == "MSR-E2E":
            take_e2e(save_path)



    def _split_generators(self, dl_manager):
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_dir = os.environ['DATASET_PATH']
        # data_dir = dl_manager.download(urls)
        self._load_path(data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if split not in self.COUNTER:
                    self.COUNTER[split] = 0
                self.COUNTER[split] += 1
                if self.config.name == "SMD":
                    # Yields examples as (key, example) tuples
                    exist = [{'text':"Hello! How can I help you?", "participant":"assistant"}] + [{"text":u['data']['utterance'],"participant":"user"}
                            if u['turn']=="driver"
                            else {"text":u['data']['utterance'],"participant":"assistant"} for u in data['dialogue']]
                    yield key, {
                        "domain": [data['scenario']['task']['intent']],
                        "dialogue_id": f"SMD--{split}--{self.COUNTER[split]}",
                        "dialogue": exist,
                    }
                elif self.config.name == "MULTIWOZ2_2":
                    exist = [[{"text":u['user utterance'],"participant":"user"},{"text":u['system response'],"participant":"assistant"}] for u in data['log']]
                    log = [{"text":"Hello! How can I help you?","participant":"assistant"}] + [x for xs in exist for x in xs]
                    yield key, {
                        "domain": ast.literal_eval(data['original dialog info'])['services'],
                        "dialogue_id": data["new dialog id"],
                        "dialogue": log,
                    }
                elif self.config.name == "META_WOZ":
                    exist = [[{"text":a,"participant":"assistant"},{"text":u,"participant":"user"}]
                             for a,u in zip(data['turns'][0:-1:2],
                                            data['turns'][1::2])]+[[{"text":data['turns'][-1],
                                "participant":"assistant"}]]
                    log = [x for xs in exist for x in xs]
                    yield key, {
                        "domain": [data['domain']],
                        "dialogue_id": data["id"],
                        "dialogue": log,
                    }
                elif self.config.name == "SCHEMA":
                    exist = [[{'text':"Hello! How can I help you?", "participant":"assistant"}]] + [[{"text":u,"participant":"user"},{"text":a,"participant":"assistant"},]
                        for u,a in zip(data['context'][0:-1:2],data['context'][1::2])] + [[{'text':data['prompt'], "participant":"user"},
                                                                                           {'text':data['target'], "participant":"assistant"}]]
                    log = [x for xs in exist for x in xs]
                    yield key, {
                        "domain": [data['service']],
                        "dialogue_id": data["gem_id"],
                        "dialogue": log,
                    }
                elif self.config.name == "TaskMaster3":
                    exist = [{'text':"Hello! How can I help you?", "participant":"assistant"}] + [{"text":u['text'],"participant":u['speaker']} for u in data['utterances']]
                    yield key, {
                        "domain": [data['vertical']],
                        "dialogue_id": data["conversation_id"],
                        "dialogue": exist,
                    }
                elif self.config.name == "MSR-E2E":
                    yield key, {
                        "domain": [data['domain']],
                        "dialogue_id": f"MSR-E2E--{split}--{self.COUNTER[split]}",
                        "dialogue": data['dialogue'],
                    }
                elif self.config.name == "Frames":
                    yield key, {
                        "domain": ['vacation'],
                        "dialogue_id": f"Frames--{split}--{self.COUNTER[split]}",
                        "dialogue": data['dialogue'],
                    }
                elif self.config.name == "WOZ":
                    exist = [[{'text':"Hello! How can I help you?", "participant":"assistant"}]] + [[{"text":u["transcript"],"participant":"user"},
                                                                                                {"text":a["system_transcript"],"participant":"assistant"}]
                                                                                                for u,a in zip(data['dialogue'][0:-1],data['dialogue'][1:])]
                    log = [x for xs in exist for x in xs]
                    yield key, {
                        "domain": ['restaurant'],
                        "dialogue_id": data['dialogue_idx'],
                        "dialogue": log,
                    }

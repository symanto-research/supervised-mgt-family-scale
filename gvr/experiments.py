# Copyright 2023 The Symanto Research Team Authors.
#
# Licensed under the CC BY-NC-SA 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from functools import partial
from itertools import product
from pathlib import Path
from typing import Dict

import pandas as pd
from datasets import concatenate_datasets
from datasets.formatting.formatting import LazyRow

from .configs import Config
from .data import load_data
from .finetune import finetune_and_evaluate


def map_label(example: LazyRow, mapping: Dict[str, str]) -> LazyRow:
    example["label"] = mapping[example["label"]]
    return example


def save_results(results: Dict[str, Dict], path: Path) -> None:
    """Saves the result of each model's classification_report in a separate json in `path`"""
    path.mkdir(parents=True, exist_ok=True)
    for model in results.keys():
        filename = f"{model}.json"
        with open(path / filename, "w") as f:
            json.dump(results[model], f, indent=4)


def save_result(result: Dict, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.json"
    with open(path / filename, "w") as f:
        json.dump(result, f, indent=4)


def model_family_experiment(
    language: str,
    family: str,
) -> None:
    """Implements the model family experiment."""
    if family not in ["params", "type"]:
        raise RuntimeError("Family must be one of 'params', 'type'.")
    if language not in ["en", "es"]:
        raise RuntimeError(
            "The data is only available in English (en) or Spanish (es)."
        )

    train = load_data(2, language, split="train")
    test = load_data(2, language, split="test")

    train = train.remove_columns(
        [x for x in train.features if x not in ["id", "text", "label"]]
    )
    test = test.remove_columns(
        [x for x in test.features if x not in ["id", "text", "label"]]
    )

    if family == "params":
        mapping = {
            "A": "approx. 1b",
            "B": "remove",
            "C": "approx. 7b",
            "D": "approx. 1b",
            "E": "approx. 7b",
            "F": "remove",
        }
        label2id = {"approx. 1b": 0, "approx. 7b": 1}
    else:
        mapping = {
            "A": "bloom",
            "B": "bloom",
            "C": "bloom",
            "D": "gpt",
            "E": "gpt",
            "F": "gpt",
        }
        label2id = {"bloom": 0, "gpt": 1}

    if family == "params":
        not_remove = set([k for k, v in mapping.items() if v != "remove"])
        train = train.filter(lambda x: x["label"] in not_remove)
        test = test.filter(lambda x: x["label"] in not_remove)

    transform_labels = partial(map_label, mapping=mapping)
    train = train.map(transform_labels)
    test = test.map(transform_labels)

    save_dirname = f"model_family/{family}/{language}/"
    save_dirpath = Path.cwd() / "results" / save_dirname

    results = {}
    for model in Config.models[language]:
        key = "-".join(model.split("/"))
        results[key] = finetune_and_evaluate(
            model, label2id, train, test, save_dirname + key
        )

        save_result(results[key], save_dirpath, key)


def detection_transference_experiment(language: str, family: str) -> None:
    """Implements the detection transference experiment"""
    if family not in ["params", "type"]:
        raise RuntimeError("Family must be one of 'params', 'type'.")
    if language not in ["en", "es"]:
        raise RuntimeError(
            "The data is only available in English (en) or Spanish (es)."
        )

    if family == "params":
        mapping = {
            "A": "1b",
            "B": "remove",
            "C": "7b",
            "D": "1b",
            "E": "7b",
            "F": "175b",
        }
        label2label = {
            "1b": "generated",
            "7b": "generated",
            "175b": "generated",
            "human": "human",
        }
    else:
        mapping = {
            "A": "bloom",
            "B": "bloom",
            "C": "bloom",
            "D": "gpt",
            "E": "gpt",
            "F": "gpt",
        }
        label2label = {
            "bloom": "generated",
            "gpt": "generated",
            "human": "human",
        }

    label2id = {"generated": 0, "human": 1}

    # Load both subtasks' data
    train_1 = load_data(1, language, split="train")
    test_1 = load_data(1, language, split="test")
    train_2 = load_data(2, language, split="train")
    test_2 = load_data(2, language, split="test")

    if family == "params":
        not_remove = set([k for k, v in mapping.items() if v != "remove"])
        train_2 = train_2.filter(lambda x: x["label"] in not_remove)
        test_2 = test_2.filter(lambda x: x["label"] in not_remove)

    # We use the human data from subtask 1 and the generated data from subtask 2
    human_label = "human"
    train_1 = train_1.filter(lambda x: x["label"] == human_label)
    test_1 = test_1.filter(lambda x: x["label"] == human_label)

    # Transform subtask 2 data
    transform_labels = partial(map_label, mapping=mapping)
    train_2 = train_2.map(transform_labels)
    test_2 = test_2.map(transform_labels)

    # Concatenate subtask 1 data so all domains are included
    train_test_1 = concatenate_datasets([train_1, test_1])
    train_test_1 = train_test_1.shuffle(seed=Config.SEED)

    # assert train_1.features.type == train_2.features.type
    # assert test_1.features.type == test_2.features.type

    save_dirname = f"detection_transfer/{family}/{language}/"
    save_dirpath = Path.cwd() / "results" / save_dirname

    results = {}

    labels = set(train_2["label"])

    for model in Config.models[language]:
        for train_label, test_label in product(labels, labels):
            current_train_2 = train_2.filter(
                lambda x: x["label"] in (train_label, human_label)
            )
            current_test_2 = test_2.filter(
                lambda x: x["label"] in (test_label, human_label)
            )

            # There's more human text per domain in subtask 1 than generated text per domain in subtask 2
            # We get same amount of human text per domain as train from top and test from bottom.
            train_2_domain_counts = pd.DataFrame(
                current_train_2["domain"]
            ).value_counts()
            test_2_domain_counts = pd.DataFrame(
                current_test_2["domain"]
            ).value_counts()
            current_train_1 = None
            current_test_1 = None
            for domain in set(current_train_2["domain"]):
                current_domain = train_test_1.filter(
                    lambda example: example["domain"] == domain
                )
                if current_train_1 == None:
                    current_train_1 = current_domain.select(
                        range(train_2_domain_counts[domain])
                    )
                else:
                    current_train_1 = concatenate_datasets(
                        [
                            current_train_1,
                            current_domain.select(
                                range(train_2_domain_counts[domain])
                            ),
                        ]
                    )

                if current_test_1 == None:
                    current_test_1 = current_domain.select(
                        range(
                            len(current_test_2) - test_2_domain_counts[domain],
                            len(current_test_2),
                        )
                    )
                else:
                    current_test_1 = concatenate_datasets(
                        [
                            current_test_1,
                            current_domain.select(
                                range(
                                    len(current_test_2)
                                    - test_2_domain_counts[domain],
                                    len(current_test_2),
                                )
                            ),
                        ]
                    )

            # only keep id text label columns
            keep_columns = ["id", "text", "label"]
            current_train_1 = current_train_1.remove_columns(
                [x for x in current_train_1.features if x not in keep_columns]
            )
            current_train_2 = current_train_2.remove_columns(
                [x for x in current_train_2.features if x not in keep_columns]
            )
            current_test_1 = current_test_1.remove_columns(
                [x for x in current_test_1.features if x not in keep_columns]
            )
            current_test_2 = current_test_2.remove_columns(
                [x for x in current_test_2.features if x not in keep_columns]
            )

            assert current_train_1.features == current_train_2.features
            assert current_test_1.features == current_test_2.features

            current_train = concatenate_datasets(
                [current_train_1, current_train_2]
            ).shuffle(Config.SEED)
            current_test = concatenate_datasets(
                [current_test_1, current_test_2]
            ).shuffle(Config.SEED)

            # Convert family labels to human vs generated so we have same label in train and test
            transform_labels = partial(map_label, mapping=label2label)
            current_train = current_train.map(transform_labels)
            current_test = current_test.map(transform_labels)

            key = f"{'-'.join(model.split('/'))}_{train_label}--{test_label}"
            results[key] = finetune_and_evaluate(
                model, label2id, current_train, current_test, save_dirname + key
            )

            save_result(results[key], save_dirpath, key)

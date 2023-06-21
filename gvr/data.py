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
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset


def load_data(subtask: int, language: str, split: str) -> Dataset:
    assert subtask in [1, 2]
    assert language in ["en", "es"]
    assert split in ["train", "test"]

    path = (
        Path.cwd()
        / "data"
        / split
        / f"subtask_{subtask}"
        / language
        / f"{split}.tsv"
    )

    df = pd.read_csv(path, sep="\t", index_col=0).reset_index(drop=True)

    return Dataset.from_pandas(df, split=split)


def gather_results(filename: Optional[str] = "results", only_f1: bool = False) -> None:
    """Gathers all results in results/ in a single big table (subj. to change)"""

    results_all = {}
    base_path = Path.cwd() / "results"

    for task in ["detection_transfer", "model_family"]:
        for family_type in ["type", "params"]:
            for language in ["en", "es"]:
                results: Dict[str, List[Any]] = {}
                path = base_path / task / family_type / language

                for idx, result_path in enumerate(sorted(path.rglob("*.json"))):
                    with open(result_path, "r") as f:
                        current_result = json.load(f)

                    unrolled_result = {}
                    # Unindent dict
                    for key in current_result.keys():
                        if type(current_result[key]) == dict:
                            for inner_key in current_result[key].keys():
                                unrolled_result[
                                    f"{key}-{inner_key}"
                                ] = current_result[key][inner_key]
                            del current_result[key][inner_key]
                        else:
                            unrolled_result[key] = current_result[key]

                    if idx == 0:
                        results["path"] = []
                        for k in unrolled_result.keys():
                            if "support" not in k and "weighted" not in k:
                                results[k] = []

                    results["path"].append(result_path.with_suffix("").name)
                    for k in results.keys():
                        if k != "path":
                            if "support" not in k and "weighted" not in k:
                                results[k].append(unrolled_result[k] * 100)

                df = pd.DataFrame.from_dict(results)
                results_all[task, family_type, language] = df

    # Postprocess
    for k in results_all.keys():  # type: ignore
        if k[0] == "detection_transfer":
            # get train and test from path
            r = results_all[k]  # type: ignore
            r.insert(0, "test", "")
            r.insert(0, "train", "")
            r.insert(0, "model", "")

            r[["model", "train--test"]] = r["path"].str.split("_", expand=True)
            r[["train", "test"]] = r["train--test"].str.split("--", expand=True)

            r.drop(["path", "train--test"], axis=1, inplace=True)

            results_all[k] = r  # type: ignore

    output = []
    for k in results_all.keys():  # type: ignore
        printable_k = f"**task: {k[0]}\tfamily: {k[1]}\tlanguage: {k[2]}**"
        output.append(printable_k)
        output.append("\n")
        cols = results_all[k].columns
        if only_f1:
            cols = [x for x in results_all[k].columns if "f1" in x or x in ["model", "path", "train", "test"]]
        output.append(
            results_all[k][cols].to_markdown(  # type: ignore
                index=False, tablefmt="github", floatfmt=".2f"
            )
        )
        output.append("\n\n")

    with open(base_path / f"{filename}.md", "w") as f:
        f.write("\n".join(output))

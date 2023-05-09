from typing import Optional

import typer

from .data import gather_results as _gather_results
from .experiments import (
    detection_transference_experiment,
    model_family_experiment,
)

app = typer.Typer()


@app.command()
def model_family_classification(
    language: str,
    family: str,
) -> None:
    """Implements model family experiment for a given subtask and language.

    Family must be defined as one of 'params' or 'type' to group models correctly.
    """
    if family not in ["params", "type"]:
        print("Family must be one of 'params', 'type'.")

    model_family_experiment(language, family)


@app.command()
def detection_transference(language: str, family: str):
    """Implements detection capability transference experiment

    Combines Subtask 1 and Subtask 2 training data.

    Family must be defined as one of 'params' or 'type' to group models correctly.
        - If 'type', trains detectors with (human, GPT) and evaluates with (human, BLOOM) and viceversa
        - If 'type', trains detectors with (human, small) and evaluates with (human, big), viceversa, and other combinations
    """
    if family not in ["params", "type"]:
        print("Family must be one of 'params', 'type'.")

    detection_transference_experiment(language, family)


@app.command()
def gather_results(filename: Optional[str] = "results", only_f1: bool = False):
    """Gathers the results of each experiment in tables in results/{filename}.md

    The extension .md is automatically added to the filename.
    """

    _gather_results(filename, only_f1)


if __name__ == "__main__":
    app()

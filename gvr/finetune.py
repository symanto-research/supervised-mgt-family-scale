import os
import random
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets.formatting.formatting import LazyRow
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

from .configs import Config


def seed_all(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def finetune(
    label2id: Dict[str, int], model_name: str, data: Dataset, save_dirname: str
):
    seed_all(Config.SEED)

    id2label = {v: k for k, v in label2id.items()}

    def preprocess_label(example: LazyRow) -> LazyRow:
        example["label"] = label2id[example["label"]]
        return example

    data = data.map(preprocess_label)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id.keys()),
        id2label=id2label,
        label2id=label2id,
    )
    model = model.to(Config.device)

    def tokenize(example: LazyRow) -> LazyRow:
        return tokenizer(example["text"])

    data = data.filter(lambda x: all([x["text"] != ""]))
    data = data.map(tokenize, batched=True, remove_columns=["text"])

    output_dir = Path.cwd() / f"checkpoints/{save_dirname}"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=Config.model2batchsize[model_name],
        num_train_epochs=5,
        weight_decay=1e-3,
        learning_rate=5e-5,
        logging_steps=20,
        save_strategy="no",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

    return output_dir


def evaluate_finetuned(
    model_path: Union[str, Path], data: Dataset, device: torch.device
):
    pipe = pipeline("text-classification", model=str(model_path), device=device)
    outputs = pipe(data["text"])
    output_labels = [output["label"] for output in outputs]
    true_labels = data["label"]

    outputs_filename = f"{model_path.name}.tsv"
    path_as_list = list(model_path.parent.parts)
    path_as_list[path_as_list.index("checkpoints")] = "outputs"
    output_dir = Path(*path_as_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / outputs_filename

    df = pd.DataFrame(
        {"id": data["id"], "text": data["text"], "hyp_label": output_labels}
    )
    df.to_csv(output_path, sep="\t", index=False)

    results = classification_report(
        y_true=true_labels, y_pred=output_labels, output_dict=True
    )

    return results


def finetune_and_evaluate(
    model_name: str,
    label2id: Dict[str, int],
    train: Dataset,
    test: Dataset,
    save_dirname: str,
):
    model_path = finetune(label2id, model_name, train, save_dirname)
    results = evaluate_finetuned(model_path, test, device=Config.device)
    return results

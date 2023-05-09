import torch


class Config:
    SEED = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {
        "en": [
            "xlm-roberta-base",
            "microsoft/deberta-v3-base",
            "bigscience/bloom-560m",
        ],
        "es": [
            "xlm-roberta-base",
            "PlanTL-GOB-ES/roberta-base-bne",
            "bigscience/bloom-560m",
        ],
    }
    model2batchsize = {
        "xlm-roberta-base": 64,
        "PlanTL-GOB-ES/roberta-base-bne": 64,
        "microsoft/deberta-v3-base": 32,
        "bigscience/bloom-560m": 64,
    }

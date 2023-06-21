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

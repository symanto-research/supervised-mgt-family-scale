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

# Set to 'true' or 'false' to run each experiment
# TRAIN_FAMILY refers to families of models trained in the same manner (e.g. GPT or BLOOM)
# PARAM_FAMILY refers to parameter scales (i.e. models with similar #params)

TRAIN_FAMILY_DETECTION=false # detect GPT vs BLOOM
PARAM_FAMILY_DETECTION=false # detect based on params

TRAIN_FAMILY_DETECTION_TRANSFERENCE=false # train with (human, GPT), evaluate with (human, BLOOM) and viceversa
PARAM_FAMILY_DETECTION_TRANSFERENCE=false # train with (human, 1b), evaluate with (human, 7b), etc.

LANGUAGES=("en" "es")

if $TRAIN_FAMILY_DETECTION; then
    FAMILY="type"
    for language in ${LANGUAGES[@]}; do
        echo "Running detection of GPT vs BLOOM in language: $language"
        CUDA_VISIBLE_DEVICES=1 python -m gvr.app model-family-classification $language $FAMILY
    done
fi

if $PARAM_FAMILY_DETECTION; then
    FAMILY="params"
    for language in ${LANGUAGES[@]}; do
        echo "Running detection of 1b vs 7b vs 175b in language: $language"
        CUDA_VISIBLE_DEVICES=1 python -m gvr.app model-family-classification $language $FAMILY
    done
fi

if $TRAIN_FAMILY_DETECTION_TRANSFERENCE; then
    FAMILY="type"
    for language in ${LANGUAGES[@]}; do
        echo "Running detection transference of GPT vs BLOOM in language: $language"
        CUDA_VISIBLE_DEVICES=1 python -m gvr.app detection-transference $language $FAMILY
    done
fi

if $PARAM_FAMILY_DETECTION_TRANSFERENCE; then
    FAMILY="params"
    for language in ${LANGUAGES[@]}; do
        echo "Running detection transference of 1b vs 7b in language: $language"
        CUDA_VISIBLE_DEVICES=1 python -m gvr.app detection-transference $language $FAMILY
    done
fi

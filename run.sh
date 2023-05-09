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

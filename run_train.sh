data=$1
model=$2

if [ -z "$data" ]; then
    echo "need data name"
    exit 1
fi

# Define models and their configuration prefixes
declare -A models_config=(
    [pidnet]="pidnet-s_b4-10k_512x512_"
    [segformer]="segformer_mit-b0_b4-10k_512x512_"
)

# Function to train a model
train_model() {
    local model_name=$1
    local cfg_prefix=$2
    python tools/train.py configs/neurocle/$model_name/$cfg_prefix$data.py --work-dir "./work_dirs/$data/$model_name"
}

# Check if specific model is given and is in the list of models
if [[ -n "$model" && -n "${models_config[$model]}" ]]; then
    train_model "$model" "${models_config[$model]}"
else
    # If no specific model or invalid model is given, iterate over all models
    for model in "${!models_config[@]}"; do
        train_model "$model" "${models_config[$model]}"
    done
fi
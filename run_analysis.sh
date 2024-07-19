data=$1
model=$2

# Define models and their configuration prefixes
declare -A models_config=(
    [pidnet]="pidnet-s_b4-10k_512x512_"
    [segformer]="segformer_mit-b0_b4-10k_512x512_"
)

# Function to train a model
getflops() {
    local model_name=$1
    local cfg_prefix=$2

    local exp_path="./work_dirs/$data/$model_name"
    local exp_config="$exp_path/$cfg_prefix$data.py"
    python tools/analysis_tools/get_flops.py $exp_config --shape 512 512 2>&1 | tee "$exp_path"_flops.log
}

benchmark() {
    local model_name=$1
    local cfg_prefix=$2

    local exp_path="./work_dirs/$data/$model_name"
    local exp_config="$exp_path/$cfg_prefix$data.py"
    local last_checkpoint=$(head -n 1 $exp_path/last_checkpoint)
    python tools/analysis_tools/benchmark.py $exp_config $last_checkpoint --log-interval 1 --repeat-times 2 2>&1 | tee "$exp_path"_benchmark.log
}

# Check if specific model is given and is in the list of models
if [[ -n "$model" && -n "${models_config[$model]}" ]]; then
    getflops "$model" "${models_config[$model]}"
    benchmark "$model" "${models_config[$model]}"
else
    # If no specific model or invalid model is given, iterate over all models
    for model in "${!models_config[@]}"; do
        getflops "$model" "${models_config[$model]}"
        benchmark "$model" "${models_config[$model]}"
    done
fi
data=$1

if [ -z "$data" ]; then
    echo "need data name"
    exit 1
fi

models=("pidnet" "segformer")
cfg_prefixs=("pidnet-s_b4-10k_512x512_" "segformer_mit-b0_b4-10k_512x512_")

for i in "${!models[@]}"; do
  model=${models[$i]}
  cfg_prefix=${cfg_prefixs[$i]}
  python tools/train.py configs/neurocle/$model/$cfg_prefix$data.py --work-dir "./work_dirs/$data/$model"
done
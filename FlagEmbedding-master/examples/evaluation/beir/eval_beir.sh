if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

dataset_names="qwen3_8b"

eval_args="\
    --eval_name beir \
    --dataset_dir /root/evaluate_data/ \
    --dataset_names $dataset_names \
    --splits test \
    --output_dir ./root/evaluate_data/results \
    --search_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 1 3 5 10 \
    --eval_output_method markdown \
    --eval_output_path /root/evaluate_data/results/beir_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_10 \
    --ignore_identical_ids True \
"

model_args="\
    --embedder_name_or_path /root/PLM/Qwen3-Embedding-0.6B \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir $HF_MODEL_CACHE \
    --reranker_max_length 1024 \
"

cmd="python -m FlagEmbedding.evaluation.beir \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd

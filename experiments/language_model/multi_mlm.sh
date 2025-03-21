#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

echo "Current Directory: $(pwd)"

cache_dir=model
max_seq_length=16 ## original: 512
data_dir=$cache_dir/fineweb/

# Ensure necessary directories exist
mkdir -p $cache_dir
mkdir -p $data_dir

function setup_wiki_data(){
    mkdir -p $cache_dir
    if [[ ! -e  $cache_dir/spm.model ]]; then
        wget -q https://huggingface.co/microsoft/mdeberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
    fi
}

setup_wiki_data

Task="MLM"

init=$1
tag=$init
case ${init,,} in
	bert-base)
	parameters=" --num_train_epochs 1 \
	--model_config bert_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--max_ngram 1 \
	--fp16 True "
		;;
	deberta-base)
	parameters=" --num_train_epochs 1 \
	--model_config deberta_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--max_ngram 3 \
	--fp16 True "
		;;
    mdeberta-base)
	parameters=" --num_train_epochs 1 \
	--model_config mdeberta_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--max_ngram 3 \
	--fp16 True "
		;;
	xlarge-v2)
	parameters=" --num_train_epochs 1 \
	--model_config deberta_xlarge.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 32 \
	--max_ngram 3 \
	--fp16 True "
		;;
	xxlarge-v2)
	parameters=" --num_train_epochs 1 \
	--warmup 10000 \
	--model_config deberta_xxlarge.json \
	--learning_rate 1e-4 \
	--train_batch_size 32 \
	--max_ngram 3 \
	--fp16 True "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "bert-base - Pretrained a bert base model with DeBERTa vocabulary (12 layers, 768 hidden size, 128k vocabulary size)"
		echo "deberta-base - Pretrained a deberta base model (12 layers, 768 hidden size, 128k vocabulary size)"
		echo "xlarge-v2 - Pretrained DeBERTa v2 model with 900M parameters (24 layers, 1536 hidden size)"
		echo "xxlarge-v2 - Pretrained DeBERTa v2 model with 1.5B parameters (48 layers, 1536 hidden size)"
		exit 0
		;;
esac

# python -m DeBERTa.apps.run --model_config config.json  \
# 	--tag $tag \
# 	--do_train \
# 	--num_training_steps 10000 \
# 	--max_seq_len $max_seq_length \
# 	--dump 1000 \
# 	--task_name $Task \
# 	--data_dir $data_dir \
# 	--vocab_path $cache_dir/spm.model \
# 	--vocab_type spm \
# 	--output_dir /tmp/ttonly/$tag/$task  $parameters

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--num_training_steps 500000 \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path $cache_dir/spm.model \
	--vocab_type spm \
	--output_dir /tmp/ttonly/$tag/$task  $parameters

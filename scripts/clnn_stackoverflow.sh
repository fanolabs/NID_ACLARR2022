#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=$1
for d in 'stackoverflow'
do
	for l in 0.1 0.5
	do
        	for k in 0.0 0.25 0.50 0.75
		do
			for s in 0 1 2 3 4 5 6 7 8 9
        		do
	    		python clnn.py \
				--data_dir data \
				--dataset $d \
				--known_cls_ratio $k \
				--labeled_ratio $l \
				--seed $s \
				--lr '1e-6' \
				--save_results_path 'clnn_outputs' \
				--view_strategy 'rtr' \
				--update_per_epoch 5 \
				--topk 500 \
				--bert_model "./pretrained_models/${d}"
    			done
		done
	done 
done

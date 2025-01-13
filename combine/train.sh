lr_list="1e-5 2e-5 5e-5 1e-4 1e-3"
task="0"
cuda=0

for lr in $lr_list
do
    output_dir="results/task${task}/combine/lr${lr}"
    mkdir -p $output_dir

    CUDA_VISIBLE_DEVICES=$cuda \
    python script/combine/combine_train.py \
        --output_dir $output_dir \
        --learning_rate $lr \
        --task $task \
        --mode s+t \
        --speech_file results/task0/xlsr53/lr2e-5/embeddings.npy \
        --text_file results/task0/bert/lr5e-5/embeddings.npy \
        > ${output_dir}/train_log.log 2>&1

    CUDA_VISIBLE_DEVICES=$cuda \
    python script/combine/infer.py $output_dir best

    python script/metric_compute.py ${output_dir}/best_predictions.json
    
done
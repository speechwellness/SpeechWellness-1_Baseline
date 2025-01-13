lr_list="2e-5 5e-5 1e-4 2e-4"
task_list="0 1 2"
cuda=0

for task in $task_list
do
    for lr in $lr_list
    do
        output_dir="results/task-${task}/xlsr53/lr${lr}"
        mkdir -p $output_dir

        CUDA_VISIBLE_DEVICES=$cuda \
        python script/audio/audio_train.py \
            --output_dir $output_dir \
            --learning_rate $lr \
            --task $task \
            --num_train_epochs 10 \
            --window_length 10 \
            --step_length 6 \
            --train_batch_size 8 \
            --eval_batch_size 8 \
            > ${output_dir}/train_log.log 2>&1

        CUDA_VISIBLE_DEVICES=$cuda \
        python script/audio/infer.py $output_dir best

        python script/metric_compute.py ${output_dir}/best_predictions.json
        

    done
done
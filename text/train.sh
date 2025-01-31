lr_list="2e-5 5e-5 1e-4 2e-4"
task_list="0 1 2"
cuda=0

for task in $task_list
do
    for lr in $lr_list
    do
        output_dir="results/task${task}/bert/lr${lr}"
        mkdir -p $output_dir
        CUDA_VISIBLE_DEVICES=$cuda \
        python script/text/text_train.py \
            --output_dir $output_dir \
            --learning_rate $lr \
            --task $task \
            > ${output_dir}/train_log.log 2>&1

        CUDA_VISIBLE_DEVICES=$cuda \
        python script/text/infer.py $output_dir
        
    done
done
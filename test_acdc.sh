device=7
task="acdc"
num_classes=4
ignore_index=4
root_path="/data/yangqinzhu/data/2023/2023-scribbleAtten/BetterScribble/WSL4MIS/data/ACDC"
fold="fold1"

model="unetformer"
test_chpt="path_to/unetformer_best_model.pth"

is_test_save=false

is_report_rw_metric=false
# arch specifics
is_tfm=true
is_sep_tfm=false
# attn crf specifics

CUDA_VISIBLE_DEVICES=${device} python -u ./code/test_main.py \
--root_path "${root_path}" \
--test_chpt "${test_chpt}" \
--is_test_save "${is_test_save}" \
--task "${task}" \
--fold "${fold}" \
--num_classes "${num_classes}" \
--ignore_index "${ignore_index}" \
--model ${model} \
--deterministic 1 \
--seed 2022 \
--is_report_rw_metric ${is_report_rw_metric} \
--is_tfm ${is_tfm} \
--is_sep_tfm ${is_sep_tfm}

 

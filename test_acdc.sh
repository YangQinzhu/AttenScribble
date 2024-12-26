device=7
task="acdc"
num_classes=4
ignore_index=4
root_path="./data/ACDC"
fold="fold1"

model="unetformer"
# test_chpt="path_to/unetformer_best_model.pth"
test_chpt="../checkpoint/e10_acdc_unetformer_spscribble_asrrw_affrrw_wcrf0.1_wacrf0.1_waff0.0_was0.0_castrue_caftrue_tftrue_stffalse_gacrfytrue_gacrfktrue_exstrue_ct0_lr0.01_alr0.01_fdfold1/unetformer_best_model.pth"

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

 

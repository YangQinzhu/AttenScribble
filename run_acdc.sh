
device=4
task="acdc"
num_classes=4
ignore_index=4
root_path="./data/ACDC"
fold="fold1"
# fold="fold2"


model="unetformer"
sup_type="scribble"
aux_seg_type="rrw"
aux_aff_type="rrw"

reg_crf_weight=0.1
attn_crf_weight=0.1
cutoff_iter=0
aux_aff_weight=0.0
aux_seg_weight=0.0

is_conf_aux_seg=true
is_conf_aux_aff=true
is_report_rw_metric=false
# arch specifics
is_tfm=true
is_sep_tfm=false
# attn crf specifics
is_y_grad_acrf=true
is_k_grad_acrf=true
is_exclude_sc=true

# base_lr=0.06
base_lr=0.01
attn_lr=0.01

exp="e10_${task}_${model}_sp${sup_type}_as${aux_seg_type}_aff${aux_aff_type}_wcrf${reg_crf_weight}_wacrf${attn_crf_weight}_waff${aux_aff_weight}_was${aux_seg_weight}_cas${is_conf_aux_seg}_caf${is_conf_aux_aff}_tf${is_tfm}_stf${is_sep_tfm}_gacrfy${is_y_grad_acrf}_gacrfk${is_k_grad_acrf}_exs${is_exclude_sc}_ct${cutoff_iter}_lr${base_lr}_alr${attn_lr}_fd${fold}"

CUDA_VISIBLE_DEVICES=${device} python -u code/train_netformer.py \
--root_path "${root_path}" \
--task "${task}" \
--checkpoint_path "../checkpoint" \
--exp "${exp}" \
--fold "${fold}" \
--num_classes "${num_classes}" \
--ignore_index "${ignore_index}" \
--sup_type "${sup_type}" \
--aux_seg_type "${aux_seg_type}" \
--aux_aff_type "${aux_aff_type}" \
--model ${model} \
--max_iterations 60000 \
--batch_size 12 \
--deterministic 1 \
--base_lr ${base_lr} \
--seed 2022 \
--reg_crf_weight ${reg_crf_weight} \
--attn_crf_weight ${attn_crf_weight} \
--aux_aff_weight ${aux_aff_weight} \
--aux_seg_weight ${aux_seg_weight} \
--is_conf_aux_seg ${is_conf_aux_seg} \
--is_conf_aux_aff ${is_conf_aux_aff} \
--eval_interval 200 \
--is_report_rw_metric ${is_report_rw_metric} \
--is_tfm ${is_tfm} \
--is_sep_tfm ${is_sep_tfm} \
--is_y_grad_acrf ${is_y_grad_acrf} \
--is_k_grad_acrf ${is_k_grad_acrf} \
--is_exclude_sc ${is_exclude_sc} \
--cutoff_iter ${cutoff_iter} \
--attn_lr ${attn_lr} 


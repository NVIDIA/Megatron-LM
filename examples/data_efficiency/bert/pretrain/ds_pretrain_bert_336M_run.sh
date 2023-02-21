###############################################################################
### Each block below is one pretraining setup. Uncomment one block to try.
###############################################################################
### Baseline cases, mostly based on Megatron-LM's BERT-Large hyperparameters,
### but with some changes (different LR schedule).
## Baseline 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million}
###############################################################################
## Baseline 703B tokens (67%):
# lr=1.5e-4
# train_iters_in_million=134e-2
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million}
###############################################################################
## Baseline 524B tokens (50%):
# lr=2e-4
# train_iters_in_million=1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million}
###############################################################################
### Curriculum learning (CL) + Random layerwise token dropping (random-LTD).
### DeepSpeed Data Efficiency's composed solution.
### BERT pretraining.
## CL+random-LTD 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# ltd_enabled="true"
# ltd_start=128
# ltd_step_in_million=2
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=96e-2
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=128
# cl_2nd_max=512
# cl_2nd_total_step_in_million=96e-2
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step_in_million} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL+random-LTD 524B tokens (50%):
# lr=2e-4
# train_iters_in_million=1
# ltd_enabled="true"
# ltd_start=128
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=48e-2
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=128
# cl_2nd_max=512
# cl_2nd_total_step_in_million=48e-2
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step_in_million} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
### Random layerwise token dropping (random-LTD).
## random-LTD 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# ltd_enabled="true"
# ltd_start=128
# ltd_step_in_million=2
# dropout=1e-1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout}
###############################################################################
## random-LTD 703B tokens (67%):
# lr=1.5e-4
# train_iters_in_million=134e-2
# ltd_enabled="true"
# ltd_start=128
# ltd_step_in_million=134e-2
# dropout=1e-1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout}
###############################################################################
## random-LTD 524B tokens (50%):
# lr=2e-4
# train_iters_in_million=1
# ltd_enabled="true"
# ltd_start=128
# ltd_step_in_million=1
# dropout=1e-1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout}
###############################################################################
### Curriculum learning (CL).
## CL vocab rarity + seqlen truncation 524B tokens (50%):
# lr=2e-4
# train_iters_in_million=1
# ltd_enabled="false"
# ltd_start=512
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=48e-2
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=128
# cl_2nd_max=512
# cl_2nd_total_step_in_million=48e-2
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step_in_million} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL vocab rarity + seqlen truncation 703B tokens (67%):
# lr=1.5e-4
# train_iters_in_million=134e-2
# ltd_enabled="false"
# ltd_start=512
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/vc_data/users/conglli/code/data_efficiency/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=64e-2
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=128
# cl_2nd_max=512
# cl_2nd_total_step_in_million=64e-2
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step_in_million} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL vocab rarity + seqlen truncation 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# ltd_enabled="false"
# ltd_start=512
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=96e-2
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=128
# cl_2nd_max=512
# cl_2nd_total_step_in_million=96e-2
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step_in_million} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL vocab rarity + seqlen reorder 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# ltd_enabled="false"
# ltd_start=512
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=1
# cl_1st_metric="seqlenvocabrarity"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/seqlen_vocab_rarity/seqlen_vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/seqlen_vocab_rarity/seqlen_vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=96e-2
# cl_1st_difficulty_step=1
# cl_1st_root=2
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root}
###############################################################################
## CL vocab rarity 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# ltd_enabled="false"
# ltd_start=512
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=1
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=96e-2
# cl_1st_difficulty_step=1
# cl_1st_root=2
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root}
###############################################################################
## CL seqlen truncation 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# ltd_enabled="false"
# ltd_start=512
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=1
# cl_1st_metric="seqlen_truncate"
# cl_1st_index_to_sample_path="dummy"
# cl_1st_index_to_metric_path="dummy"
# cl_1st_difficulty_type="value"
# cl_1st_clustering_type="single_cluster"
# cl_1st_min=128
# cl_1st_max=512
# cl_1st_total_step_in_million=96e-2
# cl_1st_difficulty_step=8
# cl_1st_root=1
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root}
###############################################################################
## CL seqlen reorder 1049B tokens (100%):
# lr=1e-4
# train_iters_in_million=2
# ltd_enabled="false"
# ltd_start=512
# ltd_step_in_million=1
# dropout=1e-1
# cl_enabled="true"
# cl_num_metric=1
# cl_1st_metric="seqlen"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/seqlen/seqlen_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/seqlen/seqlen_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="single_cluster"
# cl_1st_min=5
# cl_1st_max=100
# cl_1st_total_step_in_million=96e-2
# cl_1st_difficulty_step=8
# cl_1st_root=2
# bash ds_pretrain_bert_336M_base_script.sh ${lr} ${train_iters_in_million} \
#     ${ltd_enabled} ${ltd_start} ${ltd_step_in_million} ${dropout} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step_in_million} ${cl_1st_difficulty_step} \
#     ${cl_1st_root}
###############################################################################
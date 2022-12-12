###############################################################################
### Each block below is one pretraining setup. Uncomment one block to try.
###############################################################################
### Baseline cases, mostly based on OpenAI's GPT-3 hyperparameters, but with
### some changes (without batch size warmup, and different LR schedule).
## Baseline 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion}
###############################################################################
## Baseline 200B tokens (67%):
# lr=3.0e-4 # scaled based on train token reduction ratio
# train_tokens_in_billion=200
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion}
###############################################################################
## Baseline 150B tokens (50%):
# lr=4.0e-4
# train_tokens_in_billion=150
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion}
###############################################################################
### Curriculum learning (CL) + Random layerwise token dropping (random-LTD).
### DeepSpeed Data Efficiency's best composed solution.
## CL+random-LTD 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# ltd_enabled="true"
# ltd_start=128
# ltd_step=200000
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=110000
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=80
# cl_2nd_max=2048
# cl_2nd_total_step=110000
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL+random-LTD 150B tokens (50%):
# lr=4.0e-4
# train_tokens_in_billion=150
# ltd_enabled="true"
# ltd_start=128
# ltd_step=100000
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=55000
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=80
# cl_2nd_max=2048
# cl_2nd_total_step=55000
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
### Random layerwise token dropping (random-LTD).
## random-LTD 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# ltd_enabled="true"
# ltd_start=128
# ltd_step=200000
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step}
###############################################################################
## random-LTD 200B tokens (67%):
# lr=3.0e-4
# train_tokens_in_billion=200
# ltd_enabled="true"
# ltd_start=128
# ltd_step=133333
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step}
###############################################################################
## random-LTD 150B tokens (50%):
# lr=4.0e-4
# train_tokens_in_billion=150
# ltd_enabled="true"
# ltd_start=128
# ltd_step=100000
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step}
###############################################################################
### Curriculum learning (CL).
## CL vocab rarity + seqlen truncation 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# ltd_enabled="false"
# ltd_start=2048
# ltd_step=1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=110000
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=80
# cl_2nd_max=2048
# cl_2nd_total_step=110000
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL vocab rarity + seqlen truncation 200B tokens (67%):
# lr=3.0e-4
# train_tokens_in_billion=200
# ltd_enabled="false"
# ltd_start=2048
# ltd_step=1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=73000
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=80
# cl_2nd_max=2048
# cl_2nd_total_step=73000
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL vocab rarity + seqlen truncation 150B tokens (50%):
# lr=4.0e-4
# train_tokens_in_billion=150
# ltd_enabled="false"
# ltd_start=2048
# ltd_step=1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=55000
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=80
# cl_2nd_max=2048
# cl_2nd_total_step=55000
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL vocab rarity + seqlen reshape 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# ltd_enabled="false"
# ltd_start=2048
# ltd_step=1
# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=110000
# cl_1st_difficulty_step=1
# cl_1st_root=2
# cl_2nd_metric="seqlen_reshape"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=80
# cl_2nd_max=2048
# cl_2nd_total_step=110000
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root} ${cl_2nd_metric} ${cl_2nd_index_to_sample_path} \
#     ${cl_2nd_index_to_metric_path} ${cl_2nd_difficulty_type} \
#     ${cl_2nd_clustering_type} ${cl_2nd_min} ${cl_2nd_max} \
#     ${cl_2nd_total_step} ${cl_2nd_difficulty_step} ${cl_2nd_root}
###############################################################################
## CL vocab rarity 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# ltd_enabled="false"
# ltd_start=2048
# ltd_step=1
# cl_enabled="true"
# cl_num_metric=1
# cl_1st_metric="voc"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=110000
# cl_1st_difficulty_step=1
# cl_1st_root=2
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root}
###############################################################################
## CL seqlen truncation 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# ltd_enabled="false"
# ltd_start=2048
# ltd_step=1
# cl_enabled="true"
# cl_num_metric=1
# cl_1st_metric="seqlen_truncate"
# cl_1st_index_to_sample_path="dummy"
# cl_1st_index_to_metric_path="dummy"
# cl_1st_difficulty_type="value"
# cl_1st_clustering_type="single_cluster"
# cl_1st_min=80
# cl_1st_max=2048
# cl_1st_total_step=110000
# cl_1st_difficulty_step=8
# cl_1st_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root}
###############################################################################
## CL seqlen reshape 300B tokens (100%):
# lr=2.0e-4
# train_tokens_in_billion=300
# ltd_enabled="false"
# ltd_start=2048
# ltd_step=1
# cl_enabled="true"
# cl_num_metric=1
# cl_1st_metric="seqlen_reshape"
# cl_1st_index_to_sample_path="dummy"
# cl_1st_index_to_metric_path="dummy"
# cl_1st_difficulty_type="value"
# cl_1st_clustering_type="single_cluster"
# cl_1st_min=80
# cl_1st_max=2048
# cl_1st_total_step=110000
# cl_1st_difficulty_step=8
# cl_1st_root=1
# bash ds_pretrain_gpt_1.3B_dense_base_script.sh ${lr} \
#     ${train_tokens_in_billion} ${ltd_enabled} ${ltd_start} ${ltd_step} \
#     ${cl_enabled} ${cl_num_metric} ${cl_1st_metric} \
#     ${cl_1st_index_to_sample_path} ${cl_1st_index_to_metric_path} \
#     ${cl_1st_difficulty_type} ${cl_1st_clustering_type} ${cl_1st_min} \
#     ${cl_1st_max} ${cl_1st_total_step} ${cl_1st_difficulty_step} \
#     ${cl_1st_root}
###############################################################################
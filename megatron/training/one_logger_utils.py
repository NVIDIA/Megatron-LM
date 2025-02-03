# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import time, os

from .global_vars import get_one_logger, get_args

_one_logger_utils_version = "1.0.0-mlm"


def get_timestamp_in_ms():
    """Helper function to get timestamp in ms

    Returns:
        [int]: [timestamp in ms]
    """
    return round(time.time() * 1000.0)


def on_train_start(iteration, consumed_train_samples, train_samples, seq_length,
                   train_iters, save, async_save, log_throughput,
                   num_floating_point_operations_so_far):
    """Function will be called at the start of train function to prepare and track E2E metrics.

    Args:
        iteration (int): current iteration number
        consumed_train_samples (int): consumed sample numbers so far
        train_samples (int): total train sample number
        seq_length (int): sequence length
        train_iters (type): target iteration
        save (str): output directory to save checkpoints to
        async_save (bool): apply async checkpointing save
        log_throughput (bool): log throughput or not
        num_floating_point_operations_so_far (int): flops so far
    """
    one_logger = get_one_logger()

    if one_logger:
        with one_logger.get_context_manager():
            # Get app train loop start time
            app_train_loop_start_time = get_timestamp_in_ms()
            one_logger.store_set('app_train_loop_start_time', app_train_loop_start_time)

            # Set up initial values in store
            one_logger.store_set('iteration_start', iteration)
            one_logger.store_set('train_samples_start', consumed_train_samples)

            # Init accumulative metric values in one-logger store
            one_logger.store_set('train_iterations_time_msecs_total', 0)
            one_logger.store_set('tracked_train_iterations', iteration)
            one_logger.store_set('validation_iterations_time_msecs_total', 0)
            one_logger.store_set('tracked_validation_iterations', 0)
            one_logger.store_set('save_checkpoint_count', 0)
            one_logger.store_set('save_checkpoint_sync_time_total', 0.0)

            train_samples_target = train_samples
            train_tokens_target = seq_length * train_samples_target
            e2e_metrics = {
                'train_samples_start': consumed_train_samples,
                'train_iterations_start': iteration,
                'train_samples_target': train_samples_target,
                'train_iterations_target': train_iters,
                'train_tokens_target': train_tokens_target,
                'app_train_loop_start_time': app_train_loop_start_time,
                'is_save_checkpoint_enabled': save is not None,
                'save_checkpoint_strategy': 'async' if async_save else 'sync',
            }
            if log_throughput:
                e2e_metrics.update({
                    'train_tflop_start': float(num_floating_point_operations_so_far) / (10**12),
                })
            one_logger.log_metrics(e2e_metrics)


def _produce_e2e_metrics(log_throughput=False, throughput=None):
    """ Generate APP metrics for E2E tracking
    NOTE: always call this function after barrier call

    Args:
        log_throughput (bool, optional): if log throughput or not. Defaults to False.
        throughput (int, optional): throughput value to log. Defaults to None.

    Returns:
        dict: all E2E metrics
    """
    one_logger = get_one_logger()
    
    if one_logger:
        with one_logger.get_context_manager():
            # Unpack and assign local vars
            base_metrics = one_logger.store_get('get_e2e_base_metrics')()
            (iteration, train_duration, eval_duration, eval_iterations,
             total_flops_since_current_train_start, num_floating_point_operations_so_far,
             consumed_train_samples, world_size, seq_length) = base_metrics.values()

            iteration_start = one_logger.store_get('iteration_start')
            train_samples_start = one_logger.store_get('train_samples_start')

            train_samples = consumed_train_samples - train_samples_start
            train_iterations = iteration - iteration_start
            train_iterations_time_msecs_avg = (train_duration * 1000.0) / train_iterations
            if eval_iterations:
                validation_iterations_time_msecs_avg = (eval_duration * 1000.0) / eval_iterations
            else:
                validation_iterations_time_msecs_avg = None

            if not one_logger.store_has_key('first_logged_train_iterations_finish_time'):
                one_logger.store_set(
                    'first_logged_train_iterations_finish_time',
                    get_timestamp_in_ms()
                )

            train_tokens = train_samples * seq_length

            e2e_metrics = {
                'first_logged_train_iterations_finish_time': \
                    one_logger.store_get('first_logged_train_iterations_finish_time'),
                'train_iterations_end': iteration,
                'train_samples_end': consumed_train_samples,
                'train_iterations': train_iterations,
                'train_samples': train_samples,
                'train_iterations_time_msecs_avg': train_iterations_time_msecs_avg,
                'validation_iterations_time_total': eval_duration,
                'validation_iterations_time_msecs_avg': validation_iterations_time_msecs_avg,
                'train_tokens': train_tokens,
                'train_iterations_time_total': train_duration,
                'last_logged_train_iterations_finish_time': get_timestamp_in_ms(),
            }

            if log_throughput:
                if train_duration:
                    train_throughput_per_gpu = total_flops_since_current_train_start / (train_duration * 10**12 * world_size)
                else:
                    train_throughput_per_gpu = 0.0

                train_throughput_per_gpu_max = one_logger.store_get('train_throughput_per_gpu_max')
                if throughput:
                    train_throughput_per_gpu_max = max(throughput, train_throughput_per_gpu_max)
                    one_logger.store_set('train_throughput_per_gpu_max', train_throughput_per_gpu_max)

                throughput_metrics = {
                    'train_tflop_end': float(num_floating_point_operations_so_far) / (10**12),
                    'train_tflop': float(total_flops_since_current_train_start) / (10**12),
                    'train_throughput_per_gpu': train_throughput_per_gpu,
                    'train_throughput_per_gpu_max': train_throughput_per_gpu_max,
                }
                e2e_metrics.update(throughput_metrics)

            # Tracking minimal train/validation iteration duration metrics
            # Minimal train iteration duration
            current_train_iterations_time_msecs_total = train_duration * 1000.0
            current_train_iteration = iteration
            prev_train_iterations_time_msecs_total = one_logger.store_get('train_iterations_time_msecs_total')
            tracked_train_iterations = one_logger.store_get('tracked_train_iterations')

            if current_train_iteration > tracked_train_iterations:
                train_iterations_time_msecs = (
                    (current_train_iterations_time_msecs_total - prev_train_iterations_time_msecs_total) /
                    (current_train_iteration - tracked_train_iterations)
                )

                if not one_logger.store_has_key('train_iterations_time_msecs_min'):
                    train_iterations_time_msecs_min = train_iterations_time_msecs
                else:
                    train_iterations_time_msecs_min = min(
                        one_logger.store_get('train_iterations_time_msecs_min'),
                        train_iterations_time_msecs
                    )
                one_logger.store_set('train_iterations_time_msecs_min', train_iterations_time_msecs_min)
                one_logger.store_set('train_iterations_time_msecs_total', current_train_iterations_time_msecs_total)
                one_logger.store_set('tracked_train_iterations', current_train_iteration)

                e2e_metrics.update({
                    'train_iterations_time_msecs_min': train_iterations_time_msecs_min
                })

            # Minimal validation iteration duration
            current_validation_iterations_time_msecs_total = eval_duration * 1000.0
            current_validation_iteration = eval_iterations
            prev_validation_iterations_time_msecs_total = \
                one_logger.store_get('validation_iterations_time_msecs_total')
            tracked_validation_iterations = one_logger.store_get('tracked_validation_iterations')

            if current_validation_iteration > tracked_validation_iterations:
                validation_iterations_time_msecs = (
                    (current_validation_iterations_time_msecs_total - prev_validation_iterations_time_msecs_total) /
                    (current_validation_iteration - tracked_validation_iterations)
                )

                # Cache minimal validation iteration duration
                if not one_logger.store_has_key('validation_iterations_time_msecs_min'):
                    validation_iterations_time_msecs_min = validation_iterations_time_msecs
                else:
                    validation_iterations_time_msecs_min = min(
                        one_logger.store_get('validation_iterations_time_msecs_min'),
                        validation_iterations_time_msecs
                    )
                one_logger.store_set('validation_iterations_time_msecs_min', validation_iterations_time_msecs_min)
                one_logger.store_set('validation_iterations_time_msecs_total', current_validation_iterations_time_msecs_total)
                one_logger.store_set('tracked_validation_iterations', current_validation_iteration)

                e2e_metrics.update({
                    'validation_iterations_time_msecs_min': validation_iterations_time_msecs_min
                })
            return e2e_metrics


def track_e2e_metrics(log_throughput=False, throughput=None):
    """Track E2E application metrics with one-logger

    NOTE: the function should be called after barrier call.

    Args:
        log_throughput (bool, optional): if log throughput or not. Defaults to False.
        throughput (int, optional): throughput value to log. Defaults to None.
    """
    one_logger = get_one_logger()

    if one_logger:
        with one_logger.get_context_manager():
            e2e_metrics = _produce_e2e_metrics(log_throughput, throughput)
            one_logger.log_metrics(e2e_metrics)


def on_save_checkpoint_start(async_save):
    """Function to be called before save-checkpoint start to generate productive metrics to log after ckpt succeeds.

    Args:
        async_save (bool): apply async checkpointing save

    Returns:
        dict: productive metrics to be stored to DB after ckpt succeeds
    """
    one_logger = get_one_logger()
    
    if one_logger:
        with one_logger.get_context_manager():
            # Unpack and assign local vars
            base_metrics = one_logger.store_get('get_e2e_base_metrics')()
            (iteration, train_duration, eval_duration, eval_iterations,
             total_flops_since_current_train_start, num_floating_point_operations_so_far,
             consumed_train_samples, world_size, seq_length) = base_metrics.values()

            save_checkpoint_count = one_logger.store_get('save_checkpoint_count') + 1
            one_logger.store_set('save_checkpoint_count', save_checkpoint_count)
            one_logger.log_metrics({
                'train_iterations_save_checkpoint_end': iteration,
                'save_checkpoint_count': save_checkpoint_count,
            })
            productive_metrics = {
                'train_tflop_productive_end': float(num_floating_point_operations_so_far) / (10**12),
                'train_iterations_productive_end': iteration,
                'train_samples_productive_end': consumed_train_samples,
                'train_iterations_time_total_productive': train_duration,
                'validation_iterations_time_total_productive': eval_duration,
            }
            if async_save:
                productive_metrics.update({
                    'save_checkpoint_async_count': save_checkpoint_count,
                })
            return productive_metrics

            
def on_pretrain_start():
    """ Function to be called at the start of pretrain function to track E2E meta data
    """
    args = get_args()
    one_logger = get_one_logger()

    if one_logger:
        with one_logger.get_context_manager():
            job_name = os.environ.get('SLURM_JOB_NAME', None)
            app_tag_run_name =  job_name if not args.app_tag_run_name else args.app_tag_run_name
            app_tag_run_version = args.app_tag_run_version
            one_logger.store_set('app_tag_run_name', app_tag_run_name)
            one_logger.store_set('app_tag_run_version', app_tag_run_version)
            one_logger.store_set('train_throughput_per_gpu_max', 0.0)

            one_logger.log_metrics({
                'train_iterations_warmup': 5,
                'data_parallel_size' : args.data_parallel_size,
                'context_parallel_size': args.context_parallel_size,
                'global_batch_size': args.global_batch_size,
                'micro_batch_size': args.micro_batch_size,
                'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
                'tensor_model_parallel_size': args.tensor_model_parallel_size,
                'expert_model_parallel_size' : args.expert_model_parallel_size,
                'world_size': args.world_size,
                'model_seq_length': args.seq_length,
                'app_tag_run_name': app_tag_run_name,
                'app_tag_run_version': app_tag_run_version,
                'is_log_throughput_enabled': args.log_throughput,
                'app_run_type': 'training',
                'summary_data_schema_version': '1.0.0',
                'app_metrics_feature_tags': 'full',
                'one_logger_utils_version': _one_logger_utils_version,
            })

def track_config_flags(train_iters, skip_train, do_train, do_valid, do_test,
                           dataloader_type, retro_project_dir, retro_cyclic_train_iters):
    """Track flags about train/validation/test enablement

    Args:
        train_iters (int): target train iteration number
        skip_train (bool): flag to skip train iterations
        do_train (bool): flags to do train
        do_valid (bool): flags to do validation
        do_test (bool): flags to do test
        dataloader_type (str): dataloader type
        retro_project_dir (str): Retro project directory
        retro_cyclic_train_iters (int): iteration number for cyclic retro training
    """
    one_logger = get_one_logger()
    if one_logger:
        with one_logger.get_context_manager():
            # Update train_iters for cyclic loader
            if dataloader_type == 'cyclic' and retro_project_dir:
                assert retro_cyclic_train_iters is not None
                train_iters = retro_cyclic_train_iters
            # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
            train_enabled = train_iters and (not skip_train) and do_train and train_iters > 0
            one_logger.log_metrics({
                'is_train_iterations_enabled': train_enabled,
                'is_validation_iterations_enabled': bool(do_valid),
                'is_test_iterations_enabled': bool(do_test),
            })

def on_save_checkpoint_success(productive_metrics, async_save):
    """Function to be called after checkpointing succeeds and checkpoint is persisted for storing productive metrics

    Args:
        productive_metrics (dict): productive related E2E metrics generated at the start of save checkpoint
        async_save (bool): apply async checkpointing save
    """
    one_logger = get_one_logger()

    if one_logger:
        with one_logger.get_context_manager():
            # Accumulate train_iterations_time_total_productive for current iteration
            prod_iteration = productive_metrics['train_iterations_productive_end']

            # Log start timestamp of first iteration that was successfully checkpointed
            if not one_logger.store_has_key('first_checkpoint_success'):
                app_train_loop_start_time = one_logger.store_get('app_train_loop_start_time')
                one_logger.store_set('first_checkpoint_success', True)
                one_logger.log_metrics({
                    'first_saved_train_iterations_start_time': app_train_loop_start_time
                })

            # Handle possible out-of-order async checkpoint callbacks
            need_update = True
            if one_logger.store_has_key('iters_prod_max'):
                need_update = prod_iteration > one_logger.store_get('iters_prod_max')

            if need_update:
                # Update cache
                one_logger.store_set('iters_prod_max', prod_iteration)

                if async_save:
                    save_checkpoint_sync_time_total_productive = \
                        one_logger.store_pop(f'save_checkpoint_sync_time_total_productive:{prod_iteration}')
                    last_successful_save_checkpoint_sync_finish_time = \
                        one_logger.store_pop(f'save_checkpoint_sync_finish_time:{prod_iteration}')
                    # Update productive metrics and log to DB
                    productive_metrics.update({
                        'save_checkpoint_sync_time_total_productive': save_checkpoint_sync_time_total_productive,
                        'last_successful_save_checkpoint_sync_finish_time': last_successful_save_checkpoint_sync_finish_time
                    })
                one_logger.log_metrics(productive_metrics)


def on_save_checkpoint_end(save_checkpoint_duration, current_iteration, async_save):
    """Function to be called after checkpointing ends
    
    Args:
        save_checkpoint_duration (float): duration of current save checkpoint process
        current_iteration (int): current train iteration step number
        async_save (bool): apply async checkpointing save
    """
    one_logger = get_one_logger()
    if one_logger:
        with one_logger.get_context_manager():
            save_checkpoint_sync_finish_time = get_timestamp_in_ms()

            # Track finish timestamp of the sync part of first successful save checkpoint
            if (one_logger.store_has_key('first_checkpoint_success') 
                    and not one_logger.store_has_key('first_successful_checkpoint_end')):
                one_logger.store_set('first_successful_checkpoint_end', True)
                one_logger.log_metrics({
                    'first_successful_save_checkpoint_sync_finish_time': save_checkpoint_sync_finish_time
                })

            save_checkpoint_sync_count = one_logger.store_get('save_checkpoint_count')

            # accumulate total sync checkpointing duration
            save_checkpoint_sync_time_total = \
                one_logger.store_get('save_checkpoint_sync_time_total') + save_checkpoint_duration
            one_logger.store_set('save_checkpoint_sync_time_total', save_checkpoint_sync_time_total)

            e2e_metrics = {}
            if async_save:
                # Cache total sync checkpointing duration
                one_logger.store_set(
                    f'save_checkpoint_sync_time_total_productive:{current_iteration}',
                    save_checkpoint_sync_time_total
                )
                # Cache finish time for current iteration
                one_logger.store_set(f'save_checkpoint_sync_finish_time:{current_iteration}',
                                     save_checkpoint_sync_finish_time)
            else:
                e2e_metrics.update({
                    # Track productive total time directly for sync ckpt
                    'save_checkpoint_sync_time_total_productive': save_checkpoint_sync_time_total,
                    'last_successful_save_checkpoint_sync_finish_time': save_checkpoint_sync_finish_time,
                })

            # Tracking min & max value sync checkpointing duration
            # For the first comparison
            if not one_logger.store_has_key('save_checkpoint_sync_time_max'):
                one_logger.store_set('save_checkpoint_sync_time_max', save_checkpoint_duration)
            if not one_logger.store_has_key('save_checkpoint_sync_time_min'):
                one_logger.store_set('save_checkpoint_sync_time_min', save_checkpoint_duration)

            save_checkpoint_sync_time_max = max(
                one_logger.store_get('save_checkpoint_sync_time_max'),
                save_checkpoint_duration
            )
            save_checkpoint_sync_time_min = min(
                one_logger.store_get('save_checkpoint_sync_time_min'),
                save_checkpoint_duration
            )
            one_logger.store_set('save_checkpoint_sync_time_max', save_checkpoint_sync_time_max)
            one_logger.store_set('save_checkpoint_sync_time_min', save_checkpoint_sync_time_min)
            e2e_metrics.update({
                'save_checkpoint_sync_count': save_checkpoint_sync_count,
                'save_checkpoint_sync_time_max': save_checkpoint_sync_time_max,
                'save_checkpoint_sync_time_min': save_checkpoint_sync_time_min,
                'save_checkpoint_sync_time_total': save_checkpoint_sync_time_total,
            })
            one_logger.log_metrics(e2e_metrics)


def track_app_tag(batch_size, world_size, seq_length):
    """Track app_tag and app_tag ID

    Args:
        batch_size (int): current batch size
        world_size (int): the number of processes of current job
        seq_length (int): current sequence length
    """
    # Track app tag & app tag ID
    one_logger = get_one_logger()
    if one_logger:
        with one_logger.get_context_manager():
            app_tag_run_name = one_logger.store_get('app_tag_run_name')
            app_tag_run_version = one_logger.store_get('app_tag_run_version')
            current_app_tag = (f'{app_tag_run_name}_{app_tag_run_version}_{batch_size}'
                            f'_{world_size}_{seq_length}')
            one_logger.log_app_tag(current_app_tag)


def finish():
    """Flush E2E metrics to remote server
    """
    one_logger = get_one_logger()
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.finish()

import argparse
import os
import sys

from jet.logs.queries import Field, JETLogsQuery
from jet.utils.instance import JETInstance


def select_asset(result_obj, prefix):
    if result_obj['obj_ci']['s_job_status'] != "skipped":
        assets = result_obj.get('nested_assets', None)
        if assets is not None:
            for asset in assets:
                if asset['s_name'].startswith(prefix):
                    return asset['s_url']
    return 'not found'


def query_results(triggering_pipeline_id):
    service = JETInstance().log_service()
    query = (
        JETLogsQuery()
        .filter(Field('obj_ci.obj_upstream.l_pipeline_id') == triggering_pipeline_id)
        .filter(Field('obj_workload.s_type') == 'basic')
        .select(
            'l_exit_code', 
            'nested_assets', 
            'obj_workload.s_key', 
            'obj_workload.obj_spec', 
            'obj_ci', 
            'ts_created', 
            'obj_status.s_message',
            'obj_ci.l_job_id'
        )
        .orderby('ts_created')  # increasing (least recent in case of timestamp)
    )
    return service.query(query, flatten=False)


def dedupe_results(results):
    deduped = {}
    for result in results:
        key = result['obj_workload']['s_key']
        if key not in deduped:
            deduped[key] = result
        else:
            if result['ts_created'] > deduped[key]['ts_created']:
                deduped[key] = result

    return deduped.values()


def pretty_print_results(results, summary_jobid):
    from prettytable import PrettyTable

    exit_codes = []
    log_urls = []
    names = []
    metrics_file_urls = []
    result_message = []
    jet_log_urls = []
    for result in results:
        exit_codes.append(result.get('l_exit_code', -1))
        log_urls.append(select_asset(result, 'output_script-0.log'))
        names.append(result['obj_workload']['obj_spec']['s_name'])
        result_message.append(result['obj_status']['s_message'])
        metrics_file_urls.append(select_asset(result, 'results.json'))
        jet_log_urls.append(f"https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/{result['obj_ci']['l_job_id']}")

    # Results metrics table
    metrics_table = PrettyTable()
    metrics_table.add_column("Job Key", names, align="l")
    metrics_table.add_column("Test Result", result_message)
    metrics_table.add_column("JET Log URL", jet_log_urls)
    metrics_table.add_column("SLURM Log URL", log_urls)
    metrics_table.add_column("Results Data", metrics_file_urls, align="l")

    print(metrics_table)


def save_scripts(results, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for result in results:
        script = result['obj_workload']['obj_spec']['s_script']
        target_path = result['obj_workload']['obj_spec']['s_name'] + '.sh'
        target_path = os.path.join(save_dir, target_path)

        from textwrap import dedent
        if result['obj_workload']['obj_spec']['flat_artifacts']:
            dataset_mount = list(result['obj_workload']['obj_spec']['flat_artifacts'].keys())[0]
            content = f'''
            srun --container-image nvcr.io/nvidia/pytorch:24.01-py3 \\
                 --container-mounts "/path/to/data:{dataset_mount},/path/to/megatron-lm:/workspace/megatron-lm" \\
                 bash -c'''
            content = dedent(content)
            content += f' \'\n{script}\n\''
        else:
            content = '''
            srun --container-image nvcr.io/nvidia/pytorch:24.01-py3 \\
                 --container-mounts "/path/to/megatron-lm:/workspace/megatron-lm" \\
                 bash -c'''
            content = dedent(content)
            content += f' \'\n{script}\n\''

        with open(target_path, 'w') as script_file:
            script_file.write('#!/bin/bash')
            script_file.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pipeline_id', help="Pipeline ID for pipeline in MLM repo that triggers the JET CI")
    parser.add_argument('--download_scripts_dir', required=False,
                        help="Directory in which to save the job script.")
    parser.add_argument('--artifact_links', required=False, help="Enables job script artifact link table. Provide results summary job's ID.")
    args = parser.parse_args()

    results = query_results(args.pipeline_id)
    results = dedupe_results(results)

    if args.download_scripts_dir:
        save_scripts(results, args.download_scripts_dir)

    pretty_print_results(results, args.artifact_links)

import argparse
import os
import sys
from jet.utils.instance import JETInstance
from jet.logs.queries import JETLogsQuery, Field


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
        .select('l_exit_code', 'nested_assets', 'obj_workload.s_key', 'obj_workload.obj_spec', 'obj_ci', 'ts_created')
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

def check_exitcodes(results):
    from prettytable import PrettyTable

    exit_codes = []
    log_urls = []
    names = []
    metrics_file_urls = []
    for result in results:
        exit_codes.append(result.get('l_exit_code', -1))
        log_urls.append(select_asset(result, 'output_script-0.log'))
        names.append(result['obj_workload']['s_key'].split('basic/')[-1])
        metrics_file_urls.append(select_asset(result, 'results.json'))

    metrics_table = PrettyTable()
    metrics_table.add_column("Job Key", names)
    metrics_table.add_column("Results Data", metrics_file_urls)
    metrics_table.align["Job Key"] = 'l'
    print(metrics_table)

    table = PrettyTable()
    table.add_column("Job Key", names)
    table.add_column("Exit Code", exit_codes)
    table.add_column("Log URL", log_urls)
    table.align["Job Key"] = 'l'
    exit_codes_good = [ec == 0 for ec in exit_codes]
    if exit_codes_good == []:
        raise Exception("Can't find any jobs, something went wrong.\n" + table.get_string())
    if exit_codes_good == [] or not all(exit_codes_good):
        raise Exception("Some jobs failed to complete successfully\n" + table.get_string())
    else:
        print(table)
        print("All jobs completed successfully!")


def _download_log(url, save_dir):
    import requests
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filepath = os.path.join(save_dir, url.split('/')[-1])

    r = requests.get(url)
    if r.ok:
        with open(filepath, mode='wb') as f:
            f.write(r.content)
    else:
        print(f"WARNING: Unable to download file at {url}. Received status {r.status_code}")


def check_baselines(results):
    import pytest
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        # Download TB event logs
        for result in results:
            event_log_url = select_asset(result, 'events.out.tfevents')
            target_dir = result['obj_workload']['s_key'].split('basic/')[-1]
            target_dir = os.path.join(tmpdir, target_dir)
            _download_log(event_log_url, target_dir)

        # Run pytest on logs
        os.environ["EXPECTED_METRICS_DIR"] = "tests/functional_tests/test_results/jet"
        os.environ["LOGS_DIR"] = tmpdir
        sys.exit(pytest.main(
            ['tests/functional_tests/python_test_utils/multitest_ci_pipeline.py::TestBulkCIPipeline']))


def fetch_metrics_files(results, save_dir):
    for result in results:
        metrics_url = select_asset(result, 'results.json')
        if metrics_url is not None:
            cfg = result['obj_workload']['s_key'].split('basic/')[-1]
            target_dir = os.path.join(save_dir, cfg)
            _download_log(metrics_url, target_dir)

            with open(os.path.join(target_dir, 'results.json'), 'r') as full_results_file:
                with open(os.path.join(target_dir, cfg+'.json'), 'w') as golden_file:
                    golden_file.write(full_results_file.readlines()[-1].strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pipeline_id', help="Pipeline ID for pipeline in MLM repo that triggers the JET CI")
    parser.add_argument('--test', required=False, choices=[
                        'exit', 'metrics'], help="Check exit status of jobs with 'exit' or perf and loss with 'metrics'")
    parser.add_argument('--download_metrics_dir', help="Directory in which to save the results.json files from jobs. Will not save files if not set. Set this if you want to update golden values.")
    args = parser.parse_args()

    results = query_results(args.pipeline_id)
    results = dedupe_results(results)

    if args.download_metrics_dir:
        fetch_metrics_files(results, args.download_metrics_dir)

    if args.test == 'exit':
        check_exitcodes(results)
    elif args.test == 'metrics':
        check_baselines(results)

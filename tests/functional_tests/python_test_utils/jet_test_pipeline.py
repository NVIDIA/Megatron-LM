import argparse
import os
import sys
from jet.utils.instance import JETInstance
from jet.logs.queries import JETLogsQuery, Field


def select_asset(assets, prefix):
    for asset in assets:
        if asset['s_name'].startswith(prefix):
            return asset['s_url']


def query_results(triggering_pipeline_id):
    service = JETInstance().log_service()
    query = (
        JETLogsQuery()
        .filter(Field('obj_ci.obj_upstream.l_pipeline_id') == triggering_pipeline_id)
        .filter(Field('obj_workload.s_type') == 'recipe')
        .select('l_exit_code', 'nested_assets', 'obj_workload.s_key', 'obj_workload.obj_spec', 'ts_created')
        .orderby('ts_created')  # increasing (least recent in case of timestamp)
    )
    return service.query(query, flatten=False)


def check_exitcodes(results):
    from prettytable import PrettyTable

    exit_codes = {}
    log_urls = {}
    names = {}
    for result in results:
        key = result['obj_workload']['s_key']

        exit_codes[key] = result['l_exit_code']
        log_urls[key] = select_asset(result['nested_assets'], 'output_script-0.log')
        name = result['obj_workload']['s_key'].lstrip('recipe/')
        remove_substr = result['obj_workload']['obj_spec']['s_build'] + \
            '_' + result['obj_workload']['obj_spec']['s_scope']
        names[key] = ''.join(name.split(remove_substr))

    table = PrettyTable()
    table.add_column("Job Key", list(names.values()))
    table.add_column("Exit Code", list(exit_codes.values()))
    table.add_column("Log URL", list(log_urls.values()))
    exit_codes_good = [ec == 0 for ec in exit_codes.values()]
    if not all(exit_codes_good):
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
            event_log_url = select_asset(result['nested_assets'], 'events.out.tfevents')
            target_dir = result['obj_workload']['s_key'].lstrip('recipe/')
            target_dir = os.path.join(tmpdir, target_dir)
            _download_log(event_log_url, target_dir)

        # Run pytest on logs
        os.environ["EXPECTED_METRICS_DIR"] = "tests/functional_tests/test_results/jet"
        os.environ["LOGS_DIR"] = tmpdir
        sys.exit(pytest.main(
            ['tests/functional_tests/python_test_utils/multitest_ci_pipeline.py::TestBulkCIPipeline']))


def fetch_metrics_files(results, save_dir):
    for result in results:
        metrics_url = select_asset(result['nested_assets'], 'results.json')
        if metrics_url is not None:
            cfg = result['obj_workload']['s_key'].lstrip('recipe/')
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

    if args.download_metrics_dir:
        fetch_metrics_files(results, args.download_metrics_dir)

    if args.test == 'exit':
        check_exitcodes(results)
    elif args.test == 'metrics':
        check_baselines(results)

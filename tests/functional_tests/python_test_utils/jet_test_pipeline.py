import argparse
import os
import sys
from jet.utils.instance import JETInstance
from jet.logs.queries import JETLogsQuery, Field


def select_asset(assets, prefix):
    for asset in assets:
        if asset['s_name'].startswith(prefix):
            return asset['s_url']


def query_results(ephemeral_branch):
    service = JETInstance().log_service()
    query = (
        JETLogsQuery()
        .filter(Field('obj_workloads_registry.s_commit_ref') == ephemeral_branch)
        .filter(Field('obj_workload.s_type') == 'recipe')
        .select('l_exit_code', 'nested_assets', 'obj_workload.s_key', 'obj_workload.obj_spec')
        .orderby('-ts_created')  # decreasing (most recent in case of timestamp)
    )
    return service.query(query, flatten=False)


def check_exitcodes(results):
    from prettytable import PrettyTable

    exit_codes = []
    log_urls = []
    names = []
    for result in results:
        exit_codes.append(result['l_exit_code'])
        log_urls.append(select_asset(result['nested_assets'], 'output_script.log'))
        name = result['obj_workload']['s_key'].strip('recipe/')
        remove_substr = result['obj_workload']['obj_spec']['s_build'] + \
            '_' + result['obj_workload']['obj_spec']['s_scope']
        names.append(''.join(name.split(remove_substr)))

    table = PrettyTable()
    table.add_column("Job Key", names)
    table.add_column("Exit Code", exit_codes)
    table.add_column("Log URL", log_urls)
    exit_codes_good = [ec == 0 for ec in exit_codes]
    if not all(exit_codes_good):
        raise Exception("Some jobs failed to complete successfully\n" + table.get_string())
    else:
        print(table)
        print("All jobs completed successfully!")


def check_baselines(results):
    import requests
    import pytest
    from tempfile import TemporaryDirectory

    def download_log(url, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        filepath = os.path.join(save_dir, url.split('/')[-1])

        r = requests.get(url)
        if r.ok:
            with open(filepath, mode='wb') as f:
                f.write(r.content)
        else:
            print(f"WARNING: Unable to download file at {url}. Received status {r.status_code}")

    with TemporaryDirectory() as tmpdir:
        # Download TB event logs
        for result in results:
            event_log_url = select_asset(result['nested_assets'], 'events.out.tfevents')
            target_dir = result['obj_workload']['s_key'].lstrip('recipe/')
            target_dir = os.path.join(tmpdir, target_dir)
            download_log(event_log_url, target_dir)

        # Run pytest on logs
        os.environ["EXPECTED_METRICS_DIR"] = "tests/functional_tests/test_results/jet"
        os.environ["LOGS_DIR"] = tmpdir
        sys.exit(pytest.main(
            ['tests/functional_tests/python_test_utils/multitest_ci_pipeline.py::TestBulkCIPipeline']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'eph_branch', help="JET Workloads registry ephemeral branch created by 'jet-generate' job in this pipeline")
    parser.add_argument('--test', required=True, choices=[
                        'exit', 'metrics'], help="Check exit status of jobs with 'exit' or perf and loss with 'metrics'")
    args = parser.parse_args()

    results = query_results(args.eph_branch)

    if args.test == 'exit':
        check_exitcodes(results)
    elif args.test == 'metrics':
        check_baselines(results)

import sys
from jet.utils.instance import JETInstance
from jet.logs.queries import JETLogsQuery, Field
from prettytable import PrettyTable


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


results = query_results(sys.argv[1])

exit_codes = []
log_urls = []
names = []
for result in results:
    exit_codes.append(result['l_exit_code'])
    log_urls.append(select_asset(result['nested_assets'], 'output_script.log'))
    name = result['obj_workload']['s_key'].strip('recipe/')
    remove_substr = result['obj_workload']['obj_spec']['s_build'] + '_' + result['obj_workload']['obj_spec']['s_scope']
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

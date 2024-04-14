import argparse
import itertools
import os
import re
import yaml

SBATCH_TEMPLATE = '''
srun --container-image nvcr.io/nvidia/pytorch:23.04-py3 \\
     --container-mounts "{}:{},{}:/workspace/megatron-lm" \\
     bash -c \"
     \n{}
\"
'''


def eval_name(**globals):
    name_template = globals['name']

    to_eval = re.findall("{.*?}", name_template)
    to_eval = [x.strip('{}') for x in to_eval]
    str_to_format = re.sub("{.*?}", '{}', name_template)
    format_contents = [eval(x, globals) for x in to_eval]

    return str_to_format.format(*format_contents)


def save_script(save_dir, format, sbatch_dataset_path, sbatch_mlm_path, **globals):
    script = globals['script']

    globals['name'] = eval_name(**globals)
    globals['key'] = "basic/" + globals['name'].lower().replace('_', '-')
    globals['assets_dir'] = f"/assets/{globals['key']}"
    if format == 'sbatch' and globals['extra_args'] is not None:
        globals['extra_args'] = globals['extra_args'].replace('"', "'")

    # gather and evaluate all substitutions marked by braces in script in order of ocurrence
    to_eval = re.findall("{.*}", script)
    to_eval = [x.strip('{}') for x in to_eval]
    str_to_format = re.sub("{.*}", '{}', script)
    format_contents = [eval(x, globals) for x in to_eval]

    file_content = str_to_format.format(*format_contents)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, globals['name']+".sh"), 'w') as f:
        f.write("#!/bin/bash\n")

        if format == 'sbatch':
            dataset_mount = list(globals['artifacts'].keys())[0] if 'artifacts' in globals else "/path/to/mount/dataset"
            sbatch_content = SBATCH_TEMPLATE.format(sbatch_dataset_path, dataset_mount, sbatch_mlm_path, file_content)
            f.write(sbatch_content)
        else:
            f.write(file_content)


def main(src_yaml, save_dir, format, sbatch_dataset_path, sbatch_mlm_path):
    # load yaml
    with open(src_yaml, 'r') as f:
        raw_content = yaml.safe_load(f)

    spec_template = raw_content['spec']
    for prod in raw_content['products']:
        config = spec_template.copy()
        # expand cartesian products into list of all config overrides
        for replace in itertools.product(*prod.values()):
            # update config dict with overrides from products
            config.update({k: v for k, v in zip(prod.keys(), replace)})
            save_script(save_dir, format, sbatch_dataset_path, sbatch_mlm_path, **config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Functional tests script generator',
        description="""Generates bash or sbatch scripts
                    from yamls in this directory to run functional tests locally""")
    parser.add_argument('src_yaml', help="Yaml file in this directory from which to generate test scripts")
    parser.add_argument('--save_dir', required=False, default='./scripts',
                        help='Directory where scripts will be saved to. Defaults to ./scripts')
    parser.add_argument('--format', required=False, default='bash', choices=['bash', 'sbatch'], help="Script format")
    parser.add_argument('--sbatch-dataset-path', required=False, default='/path/to/dataset')
    parser.add_argument('--sbatch-megatronlm-path', required=False, default='/path/to/megatron-lm')
    args = parser.parse_args()

    main(args.src_yaml, args.save_dir, args.format, args.sbatch_dataset_path, args.sbatch_megatronlm_path)

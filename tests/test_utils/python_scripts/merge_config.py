import click
import yaml

@click.command()
@click.option("--model_config", type=str, help="Model config to merge")
@click.option("--base_config", type=str, help="Base config to merge")
@click.option("--runtime_config", type=str, help="Run time config to merge")
@click.option("--output_config", type=str, help="Output config to merge")
def main(model_config, base_config, runtime_config, output_config):
    
    with open(model_config, "r") as f:
        model_config = yaml.load(f)
    with open(base_config, "r") as f:
        base_config = yaml.load(f)
    with open(runtime_config, "r") as f:
        runtime_config = yaml.load(f)




if __name__ == "__main__":
    main()

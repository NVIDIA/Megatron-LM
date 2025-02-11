from argparse import ArgumentParser
import numpy as np
import pandas as pd
import wandb

def main(args):

    if args.names is not None:
        if args.runs is None:
            raise Exception("Renaming copied runs not supported when copying whole project.")
        assert len(args.names) == len(args.runs), "Number of new names must equal number of run IDs"

    args.dst_entity = args.dst_entity if args.dst_entity is not None else args.src_entity
    args.dst_project = args.dst_project if args.dst_project is not None else args.src_project
    same_project = args.src_entity == args.dst_entity and args.src_project == args.dst_project
    name_append = "-copy" if same_project and args.names is None else ""

    # Login to wandb and initialize the API
    wandb.login()
    api = wandb.Api()

    # Get the runs from the source project
    runs = api.runs(f"{args.src_entity}/{args.src_project}")

    # Iterate through the runs and copy them to the destination project
    for run in runs:
        if args.runs is not None and run.id not in args.runs:
            continue

        # Get the run history and system stream; convert to DataFrame if needed.
        history_data = run.history(samples=run.lastHistoryStep + 1)
        system_data = run.history(samples=run.lastHistoryStep + 1, stream="system")

        history = history_data if isinstance(history_data, pd.DataFrame) else pd.DataFrame(history_data)
        system = system_data if isinstance(system_data, pd.DataFrame) else pd.DataFrame(system_data)

        # Join system stream to history if system data is available
        if not system.empty:
            history = history.join(system, rsuffix="_system")

        files = run.files()

        # Determine the new run name
        name = run.name if args.names is None else args.names[args.runs.index(run.id)]

        # Create a new run in the destination project
        new_run = wandb.init(
            project=args.dst_project,
            entity=args.dst_entity,
            config=run.config,
            name=name + name_append,
            resume="allow"
        )

        # Log the history to the new run
        for index, row in history.iterrows():
            # Only log values that are not NaN
            log_data = {k: v for k, v in row.to_dict().items() 
                        if v is None or not (v == "NaN" or (isinstance(v, float) and np.isnan(v)))}
            new_run.log(log_data)

        # Upload the files to the new run
        for file in files:
            file.download(replace=True)
            new_run.save(file.name, policy="now")

        new_run.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Copies one or all of the runs in a wandb project to another.")
    parser.add_argument("-se", "--src-entity", type=str, help="Source wandb entity name.")
    parser.add_argument("-sp", "--src-project", type=str, help="Name of the wandb project.")
    parser.add_argument("-de", "--dst-entity", type=str, help="Destination wandb entity name.")
    parser.add_argument("-dp", "--dst-project", type=str, help="Name of destination wandb project.")
    parser.add_argument("-r", "--runs", nargs="*", type=str, help="List of run IDs to copy. If None, all runs will be copied.")
    parser.add_argument("-n", "--names", nargs="*", type=str, default=None, help="List of new names for copied runs (optional).")

    main(parser.parse_args())

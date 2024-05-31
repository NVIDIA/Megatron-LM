import argparse
import os
import sys

import torch

# Add megatron to the path.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)


def combine(input_files, module_prefixes, output_files):
    num_inputs_per_output = int(len(input_files) / len(output_files))

    for output_idx, output_file in enumerate(output_files):
        combined_state_dict = None

        lb = output_idx * num_inputs_per_output
        ub = (output_idx + 1) * num_inputs_per_output
        current_input_files = input_files[lb:ub]
        current_module_prefixes = module_prefixes[lb:ub]

        for i, (input_file, module_prefix) in enumerate(
            zip(current_input_files, current_module_prefixes)
        ):
            # initialize the combined state dict using the first provided input file
            current_state_dict = torch.load(input_file)
            if i == 0:
                combined_state_dict = current_state_dict.copy()
                combined_state_dict["model"] = dict()

            # copy model state dict and prefix names with the given module keys.
            for k, v in current_state_dict["model"].items():
                combined_state_dict["model"]["%s.%s" % (module_prefix, k)] = v

        torch.save(combined_state_dict, output_file)
        print("saved:", output_file)

    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Combine multiple state dicts into a single state dict.
The combined state dict is first initialized by taking a copy of the first provided input state dict.
To avoid conflicts in model parameter names, a prefix must be provided for each input file.
Model parameter names will be renamed from <original name> to <model prefix>.<original name>.


Example usage:
python combine_state_dicts.py --input language_model.pt vision_model.pt --prefixes language_model vision_model --output multimodal.pt
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", nargs="*", required=True, help="paths to input state dict files")
    parser.add_argument(
        "--prefixes",
        nargs="*",
        required=True,
        help="prefixes to use with each input model's parameters",
    )
    parser.add_argument(
        "--output", nargs="*", required=True, help="path(s) to output state dict file"
    )

    args = parser.parse_args()

    assert len(args.input) > 1, "must provide more than 1 input model to combine"
    assert len(args.input) == len(args.prefixes), "each input model must have a corresponding key"
    assert (
        len(args.input) % len(args.output) == 0
    ), "each output file must use the same number of input files"

    combine(args.input, args.prefixes, args.output)

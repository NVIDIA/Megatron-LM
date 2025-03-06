import os
import re
import glob
from typing import NamedTuple
import pandas as pd

class ExperimentSettings(NamedTuple):
    method: str
    vocab_size: int
    seq_len: int

    @property
    def vocab_size_k(self) -> int:
        return self.vocab_size // 1000

    @property 
    def seq_len_k(self) -> int:
        return self.seq_len // 1024

class ExperimentResults(NamedTuple):
    settings: ExperimentSettings
    peak_memories: list[float]  # Memory for each GPU
    iteration_time: float  # In ms

def calculate_mfu(result: ExperimentResults):
    settings = result.settings
    b = 128
    s = settings.seq_len
    h = 3072
    fh = 12288
    v = settings.vocab_size
    l = 32
    flop = 3 * (2*b*s*h*v + l * (8*b*s*h*h + 4*b*s*s*h + 4*b*s*h*fh))

    pp = 8
    tflop_per_device = flop / 1000000000000 / pp
    device_tflops = 312  # A100
    tflops = tflop_per_device / (result.iteration_time / 1000)
    mfu = tflops / device_tflops
    return mfu

def parse_case_name(filename: str) -> ExperimentSettings:
    # Parse case name like "fullpp_<method>_dp1_pp8_tp1_v<vocab_size>k_s<seq_len>k"
    pattern = r'fullpp_(\w+)_dp1_pp8_tp1_v(\d+)k_s(\d+)k'
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid case name format: {filename}")

    method, vocab_size, seq_len = match.groups()
    vocab_size = int(vocab_size)
    seq_len = int(seq_len)
    return ExperimentSettings(method, vocab_size * 1000, seq_len * 1024)

def extract_memory_values(log_file: str) -> list[float]:
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract all memory values using the same pattern as the bash script
    # memory_pattern = r'memory \(MB\).*?max allocated: (\d+\.?\d*)'
    memory_pattern = r'memory \(MB\).*?max reserved: (\d+\.?\d*)'
    memory_values = [float(x) for x in re.findall(memory_pattern, content)]
    
    return memory_values

def extract_iteration_time(log_file: str) -> float:
    with open(log_file, 'r') as f:
        content = f.read()

    # Find iteration 10's timing information
    time_pattern = r'iteration.*?10/.*?elapsed time per iteration \(ms\): (\d+\.?\d*)'
    match = re.search(time_pattern, content)
    if not match:
        raise ValueError(f"Could not find iteration 10 timing in {log_file}")

    return float(match.group(1))

def analyze_logs(log_dir: str = "./full-logs") -> list[ExperimentResults]:
    results = []

    # Find all stdout.log files in subdirectories
    for log_path in glob.glob(f"{log_dir}/*/stdout.log"):
        case_name = os.path.basename(os.path.dirname(log_path))

        try:
            settings = parse_case_name(case_name)
            peak_memories = extract_memory_values(log_path)
            iteration_time = extract_iteration_time(log_path)

            results.append(ExperimentResults(settings, peak_memories, iteration_time))
        except Exception as e:
            print(f"Warning: Failed to process {case_name}: {e}")

    return results

def display_results(results: list[ExperimentResults]):
    method_rename = {
        'base': 'baseline',
        'redis': 'redis',
        'vocab': 'vocab-2',
        'synctwice': 'vocab-1',
        'interlaced': 'interlaced',
    }

    # Convert results to DataFrame for easy display
    rows = []
    for result in results:
        mfu = calculate_mfu(result)
        mfu_str = f"{mfu * 100:.2f}%"
        row = {
            'Seq Length': result.settings.seq_len_k,
            'Vocab Size': result.settings.vocab_size_k,
            'Method': method_rename[result.settings.method],
            'MFU': mfu_str,
            # 'Iter Time (ms)': result.iteration_time,
            'Peak Memory (GB)': max(result.peak_memories) / 1024,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Define method order using the renamed values
    method_order = [method_rename[m] for m in ['base', 'redis', 'vocab', 'synctwice', 'interlaced']]

    # Sort by sequence length, vocab size, and method (with custom order)
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    df = df.sort_values(['Seq Length', 'Vocab Size', 'Method'])

    # After sorting, format the numeric columns
    df['Seq Length'] = df['Seq Length'].apply(lambda x: f"{x}k")
    df['Vocab Size'] = df['Vocab Size'].apply(lambda x: f"{x}k")

    # Display the table with separators between groups
    print("\nExperiment Results:")

    # Convert DataFrame to string
    table_str = df.to_string(index=False, float_format=lambda x: f"{x:.2f}")

    # Split into lines
    lines = table_str.split('\n')
    header = lines[0]
    data_lines = lines[1:]

    # Print header
    print(header)

    # Print data lines with separators
    prev_group = None
    for i, line in enumerate(data_lines):
        row = df.iloc[i]
        current_group = (row['Seq Length'], row['Vocab Size'])

        if prev_group is not None and current_group != prev_group:
            # Print separator line
            print('-' * len(header))

        print(line)
        prev_group = current_group

def main():
    results = analyze_logs()
    display_results(results)

if __name__ == "__main__":
    main()

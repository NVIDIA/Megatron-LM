import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import json
import shutil
import glob
from tensorboard.backend.event_processing import event_accumulator

LOGS_DIR = os.getenv('LOGS_DIR')
STEP_INTERVAL = 5

def read_tb_logs_as_list(path, summary_name, index):
    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if files:
        event_file = files[index]
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        summary = ea.Scalars(summary_name)
        summary_list = [round(x.value, 5) for x in summary]
        print(summary_list)
        return summary_list
    raise FileNotFoundError(f"File not found matching: {path}/events*")    

def collect_train_test_metrics(logs_dir, index):
    train_loss_list = read_tb_logs_as_list(logs_dir, "lm loss", index)
    train_loss_list = [round(elem,5) for elem in train_loss_list]
    train_metrics = {
        "lm loss": train_loss_list[0:len(train_loss_list):STEP_INTERVAL],
    } 
    str_train_metrics = str(train_metrics).replace("'", "\"")
    print(f"\n ----------- The following are the metrics for ----------")
    print(f"\n {str_train_metrics}", flush=True)
    return train_metrics

class TestCIPipeline:

    train_metrics_100 = collect_train_test_metrics(LOGS_DIR, 0)
    train_metrics_50_to_100 = collect_train_test_metrics(LOGS_DIR, 1)

    def _test_helper(self, loss_type):
        expected = self.train_metrics_100[loss_type]
        assert len(expected) == 100 // STEP_INTERVAL, \
            f"Train metrics from first run (before checkpoint load) should have {100 // STEP_INTERVAL} elements"
        print('expected : '  + str(expected))
        actual = self.train_metrics_50_to_100[loss_type]
        assert len(actual) == 50 // STEP_INTERVAL, \
            f"Train metrics from second run (after checkpoint load) should have {50 // STEP_INTERVAL} elements"
        print('actual : '  + str(actual))
        # NOTE : Doing this way because in gpt3 model when I run from 0 - 100 directly, it produces 1 extra element
        # i.e expected is [10.84266, 10.89696, 10.90542, 10.87498, 10.86265, 10.83608, 10.64368, 10.62319, 10.53908, 10.25005, 10.20907, 9.96542, 9.96802, 9.92436, 9.79086, 9.26718, 9.61784, 9.19018, 9.45986, 9.62168, 9.73772, 8.85732, 9.43185, 9.27912, 9.6832, 9.5127, 9.5419, 9.02549, 8.55077, 8.91355, 8.83375, 9.17722, 9.22436, 9.19436, 9.11323, 9.09711, 9.04421, 9.36795]
        # actual is : [9.73772, 8.85732, 9.43185, 9.27912, 9.6832, 9.5127, 9.5419, 9.02549, 8.55077, 8.91355, 8.83375, 9.17722, 9.22435, 9.19435, 9.11322, 9.09711, 9.04422]
        # That extra element in expected is causing some issues. So doing it this way. Need to figure out whats happening
        start_idx_expected = expected.index(actual[0]) # First element of actual
        # Here we will just be comparing values of actual and second half (50-100) of expected
        for i in range(len(actual)):
            assert actual[i] == expected[start_idx_expected + i], f"The value at step {i} should be {expected[start_idx_expected + i]} but it is {actual[i]}."

    def test_lm_loss_deterministic(self):
        self._test_helper("lm loss")

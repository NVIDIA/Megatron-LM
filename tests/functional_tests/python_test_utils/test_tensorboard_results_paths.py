# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from tests.functional_tests.python_test_utils.common import read_tb_logs_as_list


def test_reads_event_files_from_results_subdirectory(tmp_path):
    results = tmp_path / 'results'
    writer = EventFileWriter(str(results))
    writer.add_event(
        Event(step=1, summary=Summary(value=[Summary.Value(tag='lm loss', simple_value=1.25)]))
    )
    writer.close()

    metrics = read_tb_logs_as_list(str(tmp_path), train_iters=1)

    assert metrics['lm loss'].values == {1: 1.25}


def test_sorts_root_and_results_events_by_their_actual_paths(tmp_path):
    for directory, value, timestamp in [(tmp_path, 2.0, 20), (tmp_path / 'results', 1.0, 10)]:
        writer = EventFileWriter(str(directory))
        writer.add_event(
            Event(step=1, summary=Summary(value=[Summary.Value(tag='lm loss', simple_value=value)]))
        )
        writer.close()
        for event_file in directory.glob('events*tfevents*'):
            os.utime(event_file, (timestamp, timestamp))

    metrics = read_tb_logs_as_list(str(tmp_path), train_iters=1, index=0)

    assert metrics['lm loss'].values == {1: 1.0}

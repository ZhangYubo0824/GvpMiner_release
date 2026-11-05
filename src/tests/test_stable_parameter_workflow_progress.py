import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motif_stability.pipeline import progress_milestones
from motif_stability.stability import run_stability_grid
from stable_parameter_workflow.commands import CommandSpec
from stable_parameter_workflow.progress import StreamingCommandRunner, TerminalProgress


class FlushTrackingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        return super().flush()


class OverlapDetectingStream:
    def __init__(self):
        self._guard = threading.Lock()
        self.parts = []
        self.active_writers = 0
        self.max_active_writers = 0

    def write(self, value):
        with self._guard:
            self.active_writers += 1
            self.max_active_writers = max(
                self.max_active_writers, self.active_writers
            )
        time.sleep(0.01)
        self.parts.append(value)
        with self._guard:
            self.active_writers -= 1

    def flush(self):
        return None

    def getvalue(self):
        return "".join(self.parts)


def test_reporter_prints_stage_unit_percent_and_flushes():
    stream = FlushTrackingStream()
    reporter = TerminalProgress(stream=stream, clock=lambda: "14:03:21")
    reporter.emit("A", 7, 48, "STABILITY", "w30_s5 GvpK/M2 repeat 4/10")
    text = stream.getvalue()
    assert "[14:03:21] [Stage A 7/48 14.6%]" in text
    assert "[STABILITY] w30_s5 GvpK/M2 repeat 4/10" in text
    assert stream.flush_count == 1


def test_reporter_serializes_concurrent_writes():
    stream = OverlapDetectingStream()
    reporter = TerminalProgress(stream=stream, clock=lambda: "14:03:21")
    barrier = threading.Barrier(8)

    def emit(index):
        barrier.wait()
        reporter.emit("A", index, 8, "START", f"task-{index}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(emit, range(1, 9)))

    assert stream.max_active_writers == 1
    lines = stream.getvalue().splitlines()
    assert len(lines) == 8
    assert {line.rsplit(" ", 1)[-1] for line in lines} == {
        f"task-{index}" for index in range(1, 9)
    }


def test_subprocess_runner_tees_child_output_to_terminal_and_log(tmp_path):
    stream = FlushTrackingStream()
    reporter = TerminalProgress(stream=stream, clock=lambda: "00:00:00")
    spec = CommandSpec(
        "echo-test",
        (sys.executable, "-c", "print('child-output')"),
        tmp_path,
        tmp_path / "logs" / "child.log",
    )
    result = StreamingCommandRunner(reporter).run(spec, stage="A", current=1, total=1)
    assert result.returncode == 0
    assert "child-output" in stream.getvalue()
    assert "child-output" in spec.log_path.read_text(encoding="utf-8")
    assert "DONE" in stream.getvalue()


def test_stability_grid_reports_monotonic_repeat_progress():
    windows = tuple(
        {"window_id": f"w{i}", "parent_id": f"p{i}", "start": 0, "end": 4,
         "window_seq": "AAAA", "window_seq_full": "AAAA"}
        for i in range(4)
    )
    updates = []

    def clusterer(values, *, min_cluster_size, min_samples):
        return np.zeros(len(values), dtype=int)

    run_stability_grid(
        windows=windows,
        embeddings=np.arange(8, dtype=float).reshape(4, 2),
        mcs_values=(2,), min_samples_values=(1,), fraction=.75,
        repeats=3, seed=0, representation_builder=lambda values: values,
        cluster_runner=clusterer,
        progress_callback=lambda done, total, phase: updates.append((done, total, phase)),
    )
    assert updates == [
        (1, 3, "protein_subsample"),
        (2, 3, "protein_subsample"),
        (3, 3, "protein_subsample"),
    ]


def test_null_progress_is_throttled_to_at_most_101_updates():
    marks = progress_milestones(1000, max_updates=101)
    assert marks[-1] == 1000
    assert len(marks) <= 101
    assert marks == tuple(sorted(set(marks)))

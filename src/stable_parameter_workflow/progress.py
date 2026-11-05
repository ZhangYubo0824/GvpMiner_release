from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

from .commands import CommandSpec


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    elapsed_seconds: float
    log_path: Path


class TerminalProgress:
    def __init__(
        self,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.clock = clock or (lambda: datetime.now().strftime("%H:%M:%S"))
        self._lock = threading.Lock()

    def emit(
        self,
        stage: str,
        current: int,
        total: int,
        event: str,
        message: str,
    ) -> None:
        if not self.enabled:
            return
        percent = 100.0 * current / total if total else 100.0
        line = (
            f"[{self.clock()}] [Stage {stage} {current}/{total} {percent:.1f}%] "
            f"[{event}] {message}\n"
        )
        with self._lock:
            self.stream.write(line)
            self.stream.flush()


class StreamingCommandRunner:
    def __init__(self, reporter: TerminalProgress | None = None) -> None:
        self.reporter = reporter or TerminalProgress()

    def run(
        self,
        command: CommandSpec,
        *,
        stage: str,
        current: int,
        total: int,
    ) -> CommandResult:
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.reporter.emit(stage, current, total, "START", command.name)
        started = time.monotonic()
        with command.log_path.open("a", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command.argv,
                cwd=command.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                self.reporter.emit(stage, current, total, "CHILD", line.rstrip())
            returncode = process.wait()
        elapsed = time.monotonic() - started
        event = "DONE" if returncode == 0 else "FAILED"
        self.reporter.emit(
            stage, current, total, event,
            f"{command.name} exit={returncode} elapsed={elapsed:.1f}s log={command.log_path}",
        )
        return CommandResult(returncode, elapsed, command.log_path)

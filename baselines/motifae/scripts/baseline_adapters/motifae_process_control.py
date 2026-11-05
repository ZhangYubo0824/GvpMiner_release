"""Bounded, exclusive execution of an owned external process tree."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


class JobTimedOut(RuntimeError):
    pass


class ProcessCleanupError(RuntimeError):
    pass


def group_has_live_processes(group_id):
    if not isinstance(group_id,int) or group_id<=0:return False
    if os.name!='posix':return False
    proc=Path('/proc')
    if proc.is_dir():
        for directory in proc.iterdir():
            if not directory.name.isdigit():continue
            try:
                fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==group_id and fields[0]!='Z':return True
            except (OSError,ValueError,IndexError):continue
        return False
    result=subprocess.run(['ps','-axo','pgid=,stat='],capture_output=True,text=True,check=True)
    return any(len(row:=line.split())>=2 and int(row[0])==group_id and not row[1].startswith('Z')
               for line in result.stdout.splitlines())


def attempt_has_live_processes(attempt):
    """Read-only orphan detection, including a crash before PID persistence."""
    if os.name!='posix' or not Path('/proc').is_dir():return False
    attempt=Path(attempt).resolve();needle=os.fsencode(str(attempt))
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]=='Z':continue
            command=(directory/'cmdline').read_bytes()
            gibbs=(b'gibbs_cluster_' in command or b'GibbsCluster-2.0e_SA.pl' in command or b'/gibbscluster\0' in command)
            if gibbs and (needle in command or (directory/'cwd').resolve()==attempt):return True
        except (OSError,ValueError,IndexError):continue
    return False


@contextmanager
def termination_signals():
    previous={}
    if threading.current_thread() is threading.main_thread():
        def interrupt(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
        for name in ('SIGTERM','SIGHUP'):
            if hasattr(signal,name):
                number=getattr(signal,name);previous[number]=signal.getsignal(number)
                if name=='SIGHUP' and previous[number]==signal.SIG_IGN:
                    previous.pop(number);continue
                signal.signal(number,interrupt)
    try:yield
    finally:
        for number,handler in previous.items():signal.signal(number,handler)


@contextmanager
def _cleanup_signals():
    previous={}
    if threading.current_thread() is threading.main_thread():
        for name in ('SIGINT','SIGTERM','SIGHUP'):
            if hasattr(signal,name):
                number=getattr(signal,name);previous[number]=signal.getsignal(number)
                signal.signal(number,signal.SIG_IGN)
    try:yield
    finally:
        for number,handler in previous.items():signal.signal(number,handler)


@contextmanager
def _deferred_spawn_signals():
    """Defer cancellation across Popen's child-created/handle-return window.

    No signal mask is inherited by the child and no cancellation is ignored.
    Restore/deliver only inside the caller's active cleanup try/finally.
    """
    previous={};pending=[];restored=False
    def capture(signum,frame):pending.append(signum)
    def restore_and_deliver():
        nonlocal restored
        if not restored:
            for number,handler in previous.items():signal.signal(number,handler)
            restored=True
        if pending:raise KeyboardInterrupt(f'Received signal {pending[0]} during process launch')
    try:
        if threading.current_thread() is threading.main_thread():
            for name in ('SIGINT','SIGTERM','SIGHUP'):
                if hasattr(signal,name):
                    number=getattr(signal,name);handler=signal.getsignal(number)
                    if handler==signal.SIG_IGN:continue
                    previous[number]=handler;signal.signal(number,capture)
        yield restore_and_deliver
    finally:
        if not restored:
            for number,handler in previous.items():signal.signal(number,handler)


def _stop_tree_impl(process,grace_seconds=1.):
    with _cleanup_signals():
        if os.name=='posix':
            # Always examine the private group, even after the wrapper exits.
            # The sampler may still be alive in a descendant process.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            end=time.monotonic()+grace_seconds
            while time.monotonic()<end and group_has_live_processes(process.pid):
                process.poll();time.sleep(.05)
            if group_has_live_processes(process.pid):
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        elif process.poll() is None:
            # Native Gibbs is Linux/Darwin-only. This branch permits portable
            # direct-child/timeout tests without requiring additional packages.
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5,check=False)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired as exc:
            raise ProcessCleanupError(f'Could not reap owned process {process.pid}') from exc
        if os.name=='posix':
            end=time.monotonic()+1
            while group_has_live_processes(process.pid) and time.monotonic()<end:time.sleep(.05)
            if group_has_live_processes(process.pid):
                raise ProcessCleanupError(f'Owned process group {process.pid} still has live members')


def _stop_tree(process,grace_seconds=1.):
    try:
        _stop_tree_impl(process,grace_seconds)
    except ProcessCleanupError:raise
    except Exception as exc:
        raise ProcessCleanupError(f'Unable to confirm cleanup of owned process tree {process.pid}: {exc}') from exc


def run_external(command,*,cwd,stdout,stderr,timeout_seconds=300,heartbeat_seconds=30,
                 label='',on_start=None):
    for value in (timeout_seconds,heartbeat_seconds):
        if not math.isfinite(value) or value<0:raise ValueError('Runtime controls must be finite and nonnegative')
    options=dict(cwd=cwd,stdout=stdout,stderr=stderr)
    if os.name=='posix':options['start_new_session']=True
    else:options['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP
    started=time.monotonic();last_heartbeat=started
    process=None
    with _deferred_spawn_signals() as restore_signals:
        try:
            try:process=subprocess.Popen(command,**options)
            finally:restore_signals()
            if on_start is not None:on_start(process)
            while True:
                code=process.poll()
                if code is not None:return subprocess.CompletedProcess(command,code)
                now=time.monotonic();elapsed=now-started
                if timeout_seconds and elapsed>=timeout_seconds:
                    raise JobTimedOut(f'{label}: exceeded {timeout_seconds:g}s; owned process tree terminated')
                if heartbeat_seconds and now-last_heartbeat>=heartbeat_seconds:
                    print(f'[waiting] {label}: {elapsed:.0f}s elapsed; limit={timeout_seconds:g}s (0=unlimited)',flush=True)
                    last_heartbeat=now
                wait_for=.25
                if timeout_seconds:wait_for=min(wait_for,max(.001,timeout_seconds-elapsed))
                if heartbeat_seconds:wait_for=min(wait_for,max(.001,heartbeat_seconds-(now-last_heartbeat)))
                try:
                    code=process.wait(timeout=wait_for)
                    return subprocess.CompletedProcess(command,code)
                except subprocess.TimeoutExpired:pass
        finally:
            if process is not None:_stop_tree(process)


@contextmanager
def batch_lock(output_dir):
    """Kernel-held lock; keep its inode after release to prevent lock races."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    path=output/'.batch.lock'
    with path.open('a+b') as handle:
        handle.seek(0,os.SEEK_END)
        if handle.tell()==0:handle.write(b'\0');handle.flush()
        handle.seek(0)
        acquired=False
        try:
            try:
                if os.name=='posix':
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                else:
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                acquired=True
            except OSError as exc:
                raise RuntimeError(f'Batch already running for output directory: {output}') from exc
            handle.seek(1);handle.truncate()
            handle.write(json.dumps(dict(pid=os.getpid(),started_unix=time.time())).encode('utf-8'));handle.flush()
            yield
        finally:
            if acquired:
                handle.seek(0)
                if os.name=='posix':
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
                else:
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

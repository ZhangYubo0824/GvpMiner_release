import contextlib
import csv
import importlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/baseline_adapters'))


class ProcessControlTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('motifae_process_control'))
        self.api=importlib.import_module('motifae_process_control')

    def alive(self,pid):
        if os.name=='posix':
            path=Path(f'/proc/{pid}/stat')
            if path.exists():return path.read_text().rsplit(')',1)[1].split()[0]!='Z'
            try:os.kill(pid,0);return True
            except ProcessLookupError:return False
        result=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,text=True)
        return any(len(row)>1 and row[1]==str(pid) for row in csv.reader(result.stdout.splitlines()))

    def test_real_process_runs_and_emits_heartbeat(self):
        with tempfile.TemporaryDirectory() as t,contextlib.redirect_stdout(io.StringIO()) as text:
            result=self.api.run_external([sys.executable,'-c','import time;time.sleep(.4)'],
                cwd=Path(t),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                timeout_seconds=5,heartbeat_seconds=.1,label='test')
        self.assertEqual(result.returncode,0)
        self.assertIn('waiting',text.getvalue())

    def test_real_timeout_stops_direct_child(self):
        pids=[]
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(self.api.JobTimedOut):
                self.api.run_external([sys.executable,'-c','import time;time.sleep(30)'],
                    cwd=Path(t),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                    timeout_seconds=.3,heartbeat_seconds=0,on_start=lambda p:pids.append(p.pid))
        self.assertEqual(len(pids),1)
        self.assertFalse(self.alive(pids[0]))

    def test_already_finished_child_is_not_reported_as_timeout(self):
        with tempfile.TemporaryDirectory() as t:
            result=self.api.run_external([sys.executable,'-c','pass'],cwd=Path(t),
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout_seconds=.1,
                heartbeat_seconds=0,on_start=lambda p:time.sleep(.4))
        self.assertEqual(result.returncode,0)

    @unittest.skipUnless(os.name=='posix','Linux/POSIX process-group semantics')
    def test_wrapper_exit_does_not_leave_grandchild_alive(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);pidfile=root/'child.pid'
            code=("import subprocess,sys,pathlib;"
                  "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
                  f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))")
            result=self.api.run_external([sys.executable,'-c',code],cwd=root,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout_seconds=5,heartbeat_seconds=0)
            self.assertEqual(result.returncode,0)
            self.assertFalse(self.alive(int(pidfile.read_text())))

    @unittest.skipUnless(os.name=='posix','Linux/POSIX process-group semantics')
    def test_timeout_kills_grandchild_that_ignores_term(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);pidfile=root/'child.pid'
            child="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
            code=("import subprocess,sys,pathlib,time;"
                  f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);"
                  f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid));time.sleep(30)")
            with self.assertRaises(self.api.JobTimedOut):
                self.api.run_external([sys.executable,'-c',code],cwd=root,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout_seconds=1,heartbeat_seconds=0)
            self.assertFalse(self.alive(int(pidfile.read_text())))

    def test_keyboard_interrupt_after_start_cleans_child(self):
        pids=[]
        def interrupt(process):
            pids.append(process.pid)
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(KeyboardInterrupt):
                self.api.run_external([sys.executable,'-c','import time;time.sleep(30)'],cwd=Path(t),
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout_seconds=5,
                    heartbeat_seconds=0,on_start=interrupt)
        self.assertFalse(self.alive(pids[0]))

    def test_signal_during_popen_return_is_deferred_until_child_is_registered(self):
        created=[];real_popen=subprocess.Popen;injected=False
        def popen(*args,**kwargs):
            nonlocal injected
            process=real_popen(*args,**kwargs)
            if not injected:
                injected=True;created.append(process)
                handler=signal.getsignal(signal.SIGINT)
                handler(signal.SIGINT,None)
            return process
        try:
            with tempfile.TemporaryDirectory() as t,mock.patch('motifae_process_control.subprocess.Popen',side_effect=popen):
                with self.assertRaises(KeyboardInterrupt):
                    self.api.run_external([sys.executable,'-c','import time;time.sleep(30)'],cwd=Path(t),
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout_seconds=5,heartbeat_seconds=0)
            self.assertIsNotNone(created[0].poll(),'child leaked during the Popen registration window')
        finally:
            for process in created:
                if process.poll() is None:process.kill();process.wait()

    def test_cleanup_errors_are_always_classified_as_cleanup_failure(self):
        @contextlib.contextmanager
        def failure():
            raise PermissionError('injected cleanup operation failure')
            yield
        with mock.patch('motifae_process_control._cleanup_signals',failure):
            with self.assertRaises(self.api.ProcessCleanupError):
                self.api._stop_tree(mock.Mock(pid=12345))

    def test_batch_lock_rejects_second_owner_and_can_be_reacquired(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with self.api.batch_lock(root):
                with self.assertRaisesRegex(RuntimeError,'already running'):
                    with self.api.batch_lock(root):self.fail('second lock acquired')
            self.assertTrue((root/'.batch.lock').exists())
            with self.api.batch_lock(root):pass

    @unittest.skipUnless(os.name=='posix','Linux signal semantics')
    def test_sigterm_becomes_interrupt_inside_managed_context(self):
        previous=signal.getsignal(signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            with self.api.termination_signals():os.kill(os.getpid(),signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM),previous)

    def test_invalid_runtime_limits_do_not_launch_a_process(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                self.api.run_external([sys.executable,'-c','pass'],cwd=Path(t),stdout=None,stderr=None,
                                      timeout_seconds=-1,heartbeat_seconds=0)


if __name__=='__main__':unittest.main()

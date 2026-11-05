#!/usr/bin/env python3
"""Use kernel pidfds even when a Python build omits its optional wrappers.

Linux x86_64 ABI numbers: arch/x86/entry/syscalls/syscall_64.tbl,
https://github.com/torvalds/linux : 434 pidfd_open, 424 pidfd_send_signal.
No numerical-PID kill fallback is used.
"""
import argparse
import ctypes
import os
from pathlib import Path
import platform
import signal
import sys


class PidfdBackend:
    def __init__(self):
        self._open=getattr(os,'pidfd_open',None)
        self._send=getattr(signal,'pidfd_send_signal',None)
        if callable(self._open) and callable(self._send):
            self.name='python-pidfd';self._syscall=None
        else:
            if (platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64')
                    or ctypes.sizeof(ctypes.c_void_p)!=8):
                raise RuntimeError('Direct pidfd compatibility requires Linux x86_64 with a 64-bit Python')
            self.name='linux-x86_64-syscall'
            self._libc=ctypes.CDLL(None,use_errno=True)
            self._syscall=self._libc.syscall
            self._syscall.restype=ctypes.c_longlong

    def _call(self,number,*args):
        ctypes.set_errno(0)
        result=self._syscall(ctypes.c_longlong(number),*args)
        if result==-1:
            error=ctypes.get_errno()
            raise OSError(error,os.strerror(error))
        return result

    def open(self,pid):
        if not isinstance(pid,int) or pid<=0:raise ValueError('A positive process ID is required')
        if self.name=='python-pidfd':return self._open(pid,0)
        return self._call(434,ctypes.c_int(pid),ctypes.c_uint(0))

    def send(self,handle,signum):
        if self.name=='python-pidfd':return self._send(handle,signum,None,0)
        return self._call(424,ctypes.c_int(handle),ctypes.c_int(signum),ctypes.c_void_p(),ctypes.c_uint(0))


def check_backend(backend):
    handle=backend.open(os.getpid())
    try:backend.send(handle,0)
    finally:os.close(handle)
    return backend.name


def cleanup_probe(input_path,backend,*,proc_root=Path('/proc')):
    needle=os.fsencode(str(Path(input_path).resolve()))
    def matches(directory):
        args=(directory/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return bool(args and args[-1]==needle and Path(os.fsdecode(args[0])).name=='gibbs_cluster_1.2.2_DB.Linux_x86_64')
    errors=[];killed=0
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():continue
        try:
            if not matches(directory):continue
        except OSError:continue
        handle=None
        try:
            handle=backend.open(int(directory.name))
            # Recheck AFTER opening the stable process handle. A recycled PID
            # cannot redirect this handle to an unrelated process.
            if matches(directory):backend.send(handle,signal.SIGKILL);killed+=1
        except (ProcessLookupError,FileNotFoundError):pass
        except OSError as exc:errors.append(f'pid {directory.name}: {exc}')
        finally:
            if handle is not None:os.close(handle)
    if errors:raise RuntimeError('Probe cleanup failed: '+'; '.join(errors))
    return killed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('check')
    sub.add_parser('cleanup').add_argument('input_path',type=Path)
    args=parser.parse_args()
    try:
        backend=PidfdBackend()
        if args.mode=='check':
            check_backend(backend)
            print(f'pidfd preflight OK: backend={backend.name}; kernel={platform.release()}')
        else:print(f'Probe cleanup: {cleanup_probe(args.input_path,backend)} remaining sampler(s) signalled')
    except (OSError,RuntimeError) as exc:
        print(f'pidfd {args.mode} failed: {type(exc).__name__}: {exc}',file=sys.stderr)
        print('Python interfaces: os.pidfd_open='+str(callable(getattr(os,'pidfd_open',None)))+
              ', signal.pidfd_send_signal='+str(callable(getattr(signal,'pidfd_send_signal',None))),file=sys.stderr)
        raise SystemExit(2)


if __name__=='__main__':main()

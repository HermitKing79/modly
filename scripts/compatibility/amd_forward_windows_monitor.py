"""Bounded passive Windows sampling, independent of the GPU deadline thread."""
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'api'))
from services.amd_observer import windows_observation,healthy_observation
from amd_forward_diagnostics import MARKER,Trace


class WindowsMonitor:
    def __init__(self,since,send,observe=windows_observation,healthy=healthy_observation,interval=5.):
        if not 0<interval<=5.:raise ValueError('Windows checks must start at least every five seconds')
        self.since=since;self.send=send;self.observe=observe;self.healthy=healthy;self.interval=interval
        self.ready=threading.Event();self.stop_event=threading.Event();self.error=None;self.last_healthy=None
        self.samples=0;self.last_capture=None
        self.thread=threading.Thread(target=self._run,daemon=True)

    def _run(self):
        trace=Trace(self.send,origin='supervisor')
        try:
            while not self.stop_event.is_set():
                began=time.monotonic();capture=self.observe(self.since,timeout=4.)
                good=self.healthy(capture);duration=time.monotonic()-began
                trace.emit('state','windows-observer',healthy=good,query_seconds=duration,capture=capture)
                if not good or duration>4.5:raise RuntimeError('Periodic Windows observer failed or exceeded its bound')
                self.samples+=1;self.last_capture=capture;self.last_healthy=time.monotonic();self.ready.set()
                self.stop_event.wait(max(0.,self.interval-(time.monotonic()-began)))
        except BaseException as exc:self.error=repr(exc);self.ready.set()

    def start(self,timeout=5.):
        self.thread.start()
        if not self.ready.wait(timeout):raise RuntimeError('Periodic Windows observer startup timed out')
        self.require_healthy()

    def require_healthy(self):
        if self.error or self.last_healthy is None or time.monotonic()-self.last_healthy>9.:
            raise RuntimeError('Periodic Windows observer unhealthy/stale: '+str(self.error))

    def close(self,timeout=5.):
        self.stop_event.set();self.thread.join(timeout)
        return not self.thread.is_alive()

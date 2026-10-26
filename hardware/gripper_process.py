"""Agent gripper facade: serial polling and closing policy in a spawned process.

Construction is passive. No perception modules or robot controllers enter the
worker. RPC failures latch locally; neither commands nor connections are retried.
"""
from copy import deepcopy
from dataclasses import replace
import multiprocessing as mp
from multiprocessing.managers import BaseManager, BaseProxy, convert_to_error
import os
import math
import signal
import threading
import time

from .robotiq import GripperError, GripperCurrentLimitError, Robotiq2F85
from .config import validate_gripper_feedback


_SERVICES = []  # Worker-local objects, for cleanup if the owning process dies.
_METHODS = frozenset(('start_session', 'status', 'assert_healthy', 'activate', 'open',
    'close', 'move_raw', 'stop', 'close_completed', 'may_be_holding_object',
    'session_info', 'disconnect', 'contact_report'))


class _Service:
    def __init__(self, config, execute, serial_factory):
        driver_type = Robotiq2F85
        if config.contact_close_file:
            from .robotiq_contact import AgentContactGripper
            driver_type = AgentContactGripper
        self.driver = driver_type(config, execution_enabled=execute, serial_factory=serial_factory)
        _SERVICES.append(self)

    def _metadata(self):
        driver = self.driver
        with driver._io_lock:
            return dict(session={**driver.session_info(), 'worker_pid': os.getpid(),
                        'driver_type': type(driver).__name__, 'communication_mode': 'process'},
                        feedback=deepcopy(driver.last_motion_feedback))

    def call(self, method, args, kwargs):
        if method not in _METHODS:
            raise ValueError('Unsupported gripper operation')
        try:
            result = (deepcopy(getattr(self.driver, 'last_contact_close', None))
                      if method == 'contact_report' else getattr(self.driver, method)(*args, **kwargs))
            return dict(ok=True, result=result, **self._metadata())
        except Exception as error:
            return dict(ok=False, error_type=type(error).__name__, error=str(error), **self._metadata())

    def parent_lost(self):
        # Never release/retry. An in-flight closing command may be cancelled by
        # stop(), whose independent command path is already supported by driver.
        try:
            state = self.driver._last_status
            if self.driver.execution_enabled and (self.driver._command_lock.locked()
                    or (state and state.go_to and state.object_state == 0)):
                self.driver.stop()
        except Exception:
            pass
        try:
            deadline = time.monotonic()+4*self.driver.config.io_timeout_s+1
            while self.driver._command_lock.locked() and time.monotonic() < deadline:
                time.sleep(.01)
            self.driver.disconnect()
        except Exception:
            pass


def _initialize_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = mp.parent_process()
    def watch_parent():
        while parent is not None and parent.is_alive():
            threading.Event().wait(.1)
        for service in list(_SERVICES):
            service.parent_lost()
        os._exit(0)
    threading.Thread(target=watch_parent, name='gripper-owner-watch', daemon=True).start()


class _ServiceProxy(BaseProxy):
    _exposed_ = ('call',)

    def request(self, method, args, kwargs, timeout):
        # Same local manager wire format as BaseProxy._callmethod, with a bounded
        # receive. Each caller thread owns a connection, so stop can interrupt a
        # blocking close from another thread without queuing behind that close.
        if not hasattr(self._tls, 'connection'):
            self._connect()
        connection = self._tls.connection
        try:
            connection.send((self._id, 'call', (method, args, kwargs), {}))
            if not connection.poll(timeout):
                raise TimeoutError('Gripper worker RPC timed out; command outcome is unknown')
            kind, result = connection.recv()
        except BaseException:
            connection.close()
            del self._tls.connection
            raise
        if kind == '#RETURN':
            return result
        raise convert_to_error(kind, result)


class _Manager(BaseManager):
    pass


_Manager.register('GripperService', _Service, proxytype=_ServiceProxy)


class ProcessGripper:
    def __init__(self, config, *, execution_enabled=False, serial_factory=None, rpc_timeout_s=None):
        validate_gripper_feedback(config)
        if rpc_timeout_s is not None and (isinstance(rpc_timeout_s, bool)
                or not isinstance(rpc_timeout_s, (int, float))
                or not math.isfinite(rpc_timeout_s) or rpc_timeout_s <= 0):
            raise ValueError('rpc_timeout_s must be finite and positive')
        self._source_config = config
        self.config = config
        self.execution_enabled = execution_enabled
        if config.contact_close_file:
            from .robotiq_contact import load_contact_close_setup
            setup = load_contact_close_setup(config)  # Validate files before spawning or opening USB.
            self.config = replace(config, poll_interval_s=setup.poll_interval_s,
                                  keepalive_interval_s=setup.keepalive_interval_s)
        self._serial_factory = serial_factory
        self._rpc_timeout = rpc_timeout_s
        self._manager = self._proxy = None
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self._transport_error = None
        self._transport_fault = None
        self._last_session = {}
        self.last_motion_feedback = None

    def _ensure_worker(self):
        with self._lifecycle_lock:
            if self._closed:
                raise GripperError('Session is closed; call start_session() explicitly to reconnect')
            if self._transport_error:
                raise GripperError(self._transport_error+'; disconnect/reconnect explicitly; no action retried')
            if self._manager is not None:
                if not self._manager._process.is_alive():
                    self._transport_error = 'Gripper communication worker exited'
                    raise GripperError(self._transport_error+'; no action retried')
                return
            manager = _Manager(ctx=mp.get_context('spawn'))
            try:
                manager.start(initializer=_initialize_worker)
                proxy = manager.GripperService(self._source_config, self.execution_enabled, self._serial_factory)
            except BaseException:
                if getattr(manager, '_process', None) is not None:
                    manager.shutdown()
                raise
            self._manager, self._proxy = manager, proxy

    def _request(self, method, *args, allow_failed=False, **kwargs):
        if not allow_failed:
            self._ensure_worker()
        elif self._proxy is None:
            raise GripperError('No gripper worker is running')
        timeout = self._rpc_timeout or (self.config.motion_timeout_s + 4*self.config.io_timeout_s + 3
            if method in ('open', 'close', 'move_raw', 'activate')
            else max(2., self.config.startup_timeout_s + 4*self.config.io_timeout_s + 1))
        started = time.monotonic()
        try:
            packet = self._proxy.request(method, args, kwargs, timeout)
        except KeyboardInterrupt:
            # The interrupted socket is discarded. stop uses a fresh connection
            # to cancel a worker command, without opening/releasing the fingers.
            if method in ('open', 'close', 'move_raw', 'activate'):
                try:
                    self._request('stop', allow_failed=True)
                except Exception:
                    pass
            raise
        except (EOFError, OSError, TimeoutError) as error:
            self._transport_error = f'Gripper worker communication failed ({type(error).__name__})'
            self._transport_fault = self._transport_fault or dict(method=method,
                elapsed_s=time.monotonic()-started, timeout_s=timeout, unix_s=time.time(),
                error_type=type(error).__name__)
            raise GripperError(self._transport_error+'; command outcome unknown; no action retried') from error
        self._last_session = packet['session']
        self.last_motion_feedback = packet['feedback']
        if not packet['ok']:
            error_type = {'ValueError': ValueError, 'GripperCurrentLimitError': GripperCurrentLimitError}.get(
                packet['error_type'], GripperError)
            raise error_type(packet['error'])
        return packet['result']

    def start_session(self):
        with self._lifecycle_lock:
            if self._manager is None:
                self._closed = False
                self._transport_error = None
                self._transport_fault = None
        return self._request('start_session')

    def status(self):
        return self._request('status')

    def assert_healthy(self, *, require_ready=False):
        if self._manager is None:
            raise GripperError('No active gripper session; call start_session()')
        return self._request('assert_healthy', require_ready=require_ready)

    def _motion(self, method, *args, **kwargs):
        if not self.execution_enabled:
            raise GripperError('Gripper action requires execution_enabled=True')
        return self._request(method, *args, **kwargs)

    def activate(self):
        return self._motion('activate')

    def open(self, **kwargs):
        return self._motion('open', **kwargs)

    def close(self, **kwargs):
        return self._motion('close', **kwargs)

    def move_raw(self, position, **kwargs):
        return self._motion('move_raw', position, **kwargs)

    def stop(self):
        if not self.execution_enabled:
            raise GripperError('Gripper action requires execution_enabled=True')
        return self._request('stop', allow_failed=self._proxy is not None)

    def close_completed(self, state):
        return self._request('close_completed', state)

    def may_be_holding_object(self):
        return False if self._manager is None else self._request('may_be_holding_object')

    @property
    def last_contact_close(self):
        return None if self._manager is None else self._request('contact_report')

    def session_info(self):
        if self._manager is not None:
            try:
                self._request('session_info')
            except GripperError:
                pass
        alive = bool(self._manager and self._manager._process.is_alive())
        info = dict(self._last_session, communication_mode='process', worker_alive=alive,
                    worker_pid=self._manager._process.pid if self._manager else None,
                    execution_enabled=self.execution_enabled)
        info['transport_fault'] = self._transport_fault
        info.setdefault('active', False)
        info.setdefault('error', None)
        if self._transport_error:
            info.update(active=False, monitor_alive=False, error=self._transport_error)
        return info

    def disconnect(self):
        with self._lifecycle_lock:
            if self._manager is None:
                self._closed = True
                return
            if not self._manager._process.is_alive():
                self._transport_error = self._transport_error or 'Gripper communication worker exited'
            if not self._transport_error:
                try:
                    self._request('disconnect')  # Reject disconnect during an active command.
                except GripperError:
                    if not self._transport_error:
                        raise
            if self._transport_error:
                # No motion is repeated on uncertain RPC outcomes. Best-effort
                # stop only, then close the failed session; never open fingers.
                if self.execution_enabled and self._manager._process.is_alive():
                    try:
                        self._request('stop', allow_failed=True)
                    except Exception:
                        pass
            self._manager.shutdown()
            self._manager = self._proxy = None
            self._last_session.update(active=False, monitor_alive=False)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disconnect()

"""pk3 diagnostic harness (temporary): fingerprint every process-exit path.

Auto-imported by the `site` module because the megatron-rl repo root is the
first PYTHONPATH entry of RL jobs launched from this tree. Active only when
PK3_TRACE=1 AND SLURM_PROCID is set, so production runs and other users of the
tree are unaffected.

Context (pk2 adversarial event, 2026-07-20): 31/32 ranks of every packed+CP
run exit with a clean, silent exit code 1 ~70-90 s after rollout collection —
no traceback, no crash file, no excepthook. This records WHICH exit site fired
(os._exit / sys.exit / SystemExit-finalization / thread exception), with
all-thread stacks fsync'd before death, plus a 30 s flight recorder and a
host-memory sampler to test the memory-exhaustion trigger and the head-node
exemption. Absence of every Python fingerprint while slurm still reports a
clean exit implicates a native exit() beneath the interpreter.
"""

import os

if os.environ.get('PK3_TRACE') == '1' and os.environ.get('SLURM_PROCID') is not None:
    import atexit
    import faulthandler
    import sys
    import threading
    import time
    import traceback

    _RANK = os.environ.get('SLURM_PROCID')
    _DIR = os.environ.get('PK3_DIR', '/dev/shm')

    def _write(name, text, sync=True):
        for base in dict.fromkeys((_DIR, '/dev/shm')):
            try:
                fd = os.open(
                    os.path.join(base, f'{name}_r{_RANK}.log'),
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o644,
                )
                try:
                    os.write(fd, text.encode())
                    if sync:
                        os.fsync(fd)
                finally:
                    os.close(fd)
            except Exception:
                pass

    def _stacks():
        try:
            out = []
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                out.append(f'--- thread {names.get(ident, ident)} ---\n')
                out.extend(traceback.format_stack(frame))
            return ''.join(out)
        except Exception as e:
            return f'[stack collection failed: {e!r}]\n'

    _real_os_exit = os._exit

    def _traced_os_exit(code):
        _write(
            'pk3_exit',
            f'=== os._exit({code}) t={time.time():.3f} '
            f'thread={threading.current_thread().name} ===\n{_stacks()}',
        )
        _real_os_exit(code)

    os._exit = _traced_os_exit

    _real_sys_exit = sys.exit

    def _traced_sys_exit(code=None):
        _write(
            'pk3_exit',
            f'=== sys.exit({code!r}) t={time.time():.3f} '
            f'thread={threading.current_thread().name} ===\n{_stacks()}',
        )
        _real_sys_exit(code)

    sys.exit = _traced_sys_exit

    def _atexit_marker():
        _write(
            'pk3_exit',
            f'=== atexit reached (normal finalization or uncaught SystemExit; '
            f'os._exit was NOT used) t={time.time():.3f} ===\n{_stacks()}',
        )

    atexit.register(_atexit_marker)

    _prev_thread_hook = threading.excepthook

    def _thread_hook(hook_args):
        try:
            _write(
                'pk3_exit',
                f'=== threading.excepthook {hook_args.exc_type.__name__} in thread '
                f'{getattr(hook_args.thread, "name", "?")} t={time.time():.3f} ===\n'
                + ''.join(
                    traceback.format_exception(
                        hook_args.exc_type, hook_args.exc_value, hook_args.exc_traceback
                    )
                ),
            )
        except Exception:
            pass
        _prev_thread_hook(hook_args)

    threading.excepthook = _thread_hook

    _prev_unraisable = sys.unraisablehook

    def _unraisable_hook(args):
        try:
            _write(
                'pk3_exit',
                f'=== unraisable {args.exc_type.__name__}: {args.exc_value} '
                f't={time.time():.3f} ===\n',
                sync=False,
            )
        except Exception:
            pass
        _prev_unraisable(args)

    sys.unraisablehook = _unraisable_hook

    # Fatal-signal stack dumps (SIGSEGV/SIGABRT/...). Per-pid file: every python
    # process on the rank (trainer, burn daemon, gym helpers) imports this hook,
    # and a shared file interleaves dumps.
    # NOTE: no faulthandler.dump_traceback_later here — its watchdog thread walks
    # frames without the GIL and segfaulted ranks mid-Triton-JIT (pk3 attempts
    # 1+2: tasks 16/30 then 1, all during create_cuda_graphs compile). The
    # periodic flight recorder below uses sys._current_frames under the GIL.
    try:
        _fr = open(os.path.join(_DIR, f'pk3_flight_r{_RANK}_p{os.getpid()}.log'), 'a')
        faulthandler.enable(file=_fr, all_threads=True)
    except Exception:
        pass

    def _mem_sampler():
        tick = 0
        while True:
            try:
                avail = rss = '?'
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemAvailable'):
                            avail = line.split()[1]
                            break
                with open('/proc/self/status') as f:
                    for line in f:
                        if line.startswith('VmRSS'):
                            rss = line.split()[1]
                            break
                _write('pk3_mem', f'{time.time():.1f} avail_kb={avail} rss_kb={rss}\n', sync=False)
                tick += 1
                if tick % 3 == 0:
                    # GIL-safe periodic flight recorder (replaces dump_traceback_later).
                    try:
                        _fr.write(f'=== flight t={time.time():.1f} ===\n{_stacks()}')
                        _fr.flush()
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(10)

    threading.Thread(target=_mem_sampler, name='pk3-mem-sampler', daemon=True).start()

    _write('pk3_exit', f'=== pk3 trace armed t={time.time():.3f} pid={os.getpid()} ===\n')

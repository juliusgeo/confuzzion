from threading import Thread, Lock
import sys
from codetransformer import CodeTransformer, instructions, pattern, Code
from codetransformer import patterns
from time import monotonic_ns, sleep
import numpy as np
from functools import cache
from drop_gil import f

global_v = 0
global_v_2 = 0
global_lock = Lock()


def time_module_sleep():
    sleep(0)
    return True


def my_extension_sleep():
    f(0)
    return True


def no_sleeps():
    return False


class BasicClass:
    val = 0
    lock = Lock()


class SysIntervalContext(object):
    def __init__(self, interval):
        self.__interval = interval

    def __enter__(self):
        sys.setswitchinterval(self.__interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setswitchinterval(.005)


class InjectSleep(CodeTransformer):
    def __init__(self, func):
        self.__payload = list(Code.from_pyfunc(func))[:-2]

    @pattern(
        instructions.LOAD_GLOBAL,
        patterns.matchany[patterns.var],
        instructions.STORE_GLOBAL,
    )
    def entry(self, *args):
        prev = None
        for instr, lineno in zip(self.code, sorted(self.code.lno_of_instr.values())):
            if prev != lineno:
                prev = lineno
                yield from self.__payload
            yield instr


def foo(obj):
    global global_v
    x = global_v
    global_v = x+1


def foo_safe(obj):
    global global_v_2
    with global_lock:
        x = global_v_2
        global_v_2 = x + 1


def foo_object(obj):
    x = obj.val
    obj.val = x + 1


def foo_object_safe(obj):
    with obj.lock:
        x = obj.val
        obj.val = x + 1


@cache
def transform_func(func, inject):
    if inject():
        return InjectSleep(inject)(func)
    return func


def run(n_workers, fun, args, inject, interval=.005):
    funcs = [transform_func(fun, inject) for _ in range(n_workers)]
    pool = [Thread(target=func, args=args) for func in funcs]
    with SysIntervalContext(interval):
        beg = monotonic_ns()
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        end = monotonic_ns()
    return end-beg


if __name__ == '__main__':
    n_threads = 4
    expected = n_threads
    times = []
    errors = []
    labels = []
    for inject in [my_extension_sleep, time_module_sleep, no_sleeps]:
        for interval in [.0000000001, .005]:
            for func in [foo, foo_safe, foo_object, foo_object_safe]:
                labels.append(f"({interval}, {inject.__name__}, {func.__name__})")
                avg_time = []
                avg_error = []
                for _ in range(1000):
                    global_v = 0
                    global_v_2 = 0
                    thread_obj = BasicClass()
                    avg_time.append((run(n_threads, func, [thread_obj], inject=inject, interval=interval)))
                    avg_error.append(100*(abs(expected-max(global_v, global_v_2, thread_obj.val)))/expected)
                times.append(np.mean(avg_time, axis=0))
                errors.append(np.mean(avg_error, axis=0))


print("{:<50} {:<10} {:<15}".format('COMBO', 'ERROR (%)', 'TIME (ms)'))

for row in zip(labels, errors, times):
    print("{:<50} {:<10} {:<15}".format(*row))

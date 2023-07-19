from threading import Thread, Lock
import sys
from time import monotonic_ns, sleep
from functools import cache

from codetransformer import CodeTransformer, instructions, pattern, Code, patterns
import numpy as np
from drop_gil import f, f2


class SysIntervalContext(object):
    def __init__(self, interval):
        self.__interval = interval

    def __enter__(self):
        temp = sys.getswitchinterval()
        sys.setswitchinterval(self.__interval)
        self.__interval = temp

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setswitchinterval(self.__interval)


class InjectSleep(CodeTransformer):
    def __init__(self, func):
        self.__payload = list(Code.from_pyfunc(func))[:-2]

    @pattern(
        instructions.LOAD_GLOBAL | instructions.LOAD_FAST,
        patterns.matchany[patterns.var],
        instructions.STORE_GLOBAL | instructions.STORE_ATTR,
    )
    def entry(self, *args):
        for instr in args:
            yield from self.__payload
            yield instr

global_v = 0
global_v_2 = 0
global_lock = Lock()


class BasicClass:
    val = 0
    lock = Lock()


def time_module_sleep():
    sleep(0)
    return True


def my_extension_sleep():
    f(0)
    return True


def my_extension_sleep_random():
    f2(0)
    return True


def no_sleeps():
    return False


def foo(obj):
    global global_v
    global_v = global_v+1


def foo_safe(obj):
    global global_v_2
    with global_lock:
        global_v_2 = global_v_2 + 1


def foo_object(obj):
    obj.val = obj.val + 1


def foo_object_safe(obj):
    with obj.lock:
        obj.val = obj.val + 1

@cache
def transform_func(func, inject):
    if inject():
        return InjectSleep(inject)(func)
    return func


def run_threads(n_workers, fun, args, inject, interval=.005):
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


def run_scenario(n_threads, inject, interval, func, num_iter):
    global global_v, global_v_2, global_lock
    expected = n_threads
    label = f"({interval:.3e}, {func.__name__}, {inject.__name__})"
    avg_time = []
    avg_error = []
    for _ in range(num_iter):
        global_v = 0
        global_v_2 = 0
        thread_obj = BasicClass()
        avg_time.append((run_threads(n_threads, func, [thread_obj], inject=inject, interval=interval)) / 10 ** 3)
        avg_error.append(100 * (abs(expected - max(global_v, global_v_2, thread_obj.val))) / expected)
    return label, np.mean(avg_time, axis=0), np.mean(avg_error, axis=0)


def get_func_from_name(name):
    return getattr(sys.modules[__name__], name)


def output_human_table(labels, errors, times):
    print("{:<80} {:<15} {:<15}".format('COMBO', 'ERROR (%)', 'TIME (ms)'))
    for row in sorted(zip(labels, errors, times), key=lambda x: x[1]):
        print("{:<80} {:<15} {:<15}".format(*row))


def output_markdown_table(labels, errors, times):
    print("|{}|{}|{}|".format('COMBO', 'ERROR (%)', 'TIME (ms)'))
    print("|{}|{}|{}|".format('-------', '--------', '---------'))
    for row in sorted(zip(labels, np.around(errors, decimals=3), np.around(times, decimals=3)), key=lambda x: x[1]):
        print("|{}|{}|{}|".format(*row))

if __name__ == '__main__':
    n_threads = 8
    num_iter = 1000
    labels, times, errors = [], [], []
    if len(sys.argv) > 1:
        inject, interval, func, num_iter = get_func_from_name(sys.argv[1]), float(sys.argv[2]), get_func_from_name(
            sys.argv[3]), int(sys.argv[4])
        print(run_scenario(n_threads, inject, interval, func, num_iter))

    for inject in [no_sleeps, my_extension_sleep, time_module_sleep, my_extension_sleep_random]:
        for interval in [1/2**128, .005]:
            for func in [foo, foo_object, foo_safe, foo_object_safe]:
                label, avg_time, avg_error = run_scenario(n_threads, inject, interval, func, num_iter)
                labels.append(label)
                times.append(avg_time)
                errors.append(avg_error)
    output_human_table(labels, errors, times)




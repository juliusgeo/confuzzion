# Anomalous Sea Snake or: How I Learned to Stop Worrying and Love the GIL

If you've played around with the `threading` module in Python, you have probably heard of this thing unique to Python (well, just some implementations of Python to be exact, but I will not make that distinction) called the Global Interpreter Lock. It is basically a lock (semaphore underneath, but again, will not make that distinction) that controls access to the Python interpreter. This means that to execute any Python interpreter operation, you must hold this lock. In this article I will show you how to utilize the GIL combined with a few other aspects of Python's threading model to build a convincing case for my approach to concurrency fuzzing. 

So the first question is: how does scheduling work for Python threads?
Well, the answer to that depends on what OS you're running, as Python basically leaves it up to your kernel's thread scheduler.

> "Also, which thread becomes scheduled ... is the operating system’s decision. The interpreter doesn’t have its own scheduler."[1]

However, you might be thinking to yourself--if you need the GIL to run Python code, then can't we control who runs code by controlling the GIL instead? First, I read [2] and from that learned the GIL is dropped automatically every 100 interpreter instructions (or ticks), and then I also read the documentation for `sys.setswitchinterval()` [1] which told me that it sets the minimum time that the interpreter will wait before trying to take back the GIL and handling signals, etc. I then read this [3] which gave me my fundamental idea: just get the GIL to be dropped as often as possible. It makes sense,
at least on a surface level: if we are constantly switching threads as much as possible it's more likely those threads will overlap on any 
unprotected code paths that may exist.

So that leads to the first actual code. We can do something like this:

```python
...
global x
...
def foo():
    global x
    y = x
    x = y+1
...
```

Now all we must do is spin up our threads, execute them, measure the error, and do this over and over again, and see if we have increased the chance 
of introducing a data race. So let's try running this scenario 1 hundred times! This is what I get on my Apple M1 processor when I use my test bench program:

```shell
COMBO                                              ERROR (%)  TIME (ms)      
(1e-14, foo)                                       0.0             1498591.69     
python main.py no_sleeps .00000000000001 foo 100  0.67s user 1.42s system 422% cpu 0.495 total
```

Darn, ok. That doesn't seem to be working. More iterations?

```shell
COMBO                                              ERROR (%)  TIME (ms)      
(1e-14, foo)                                       0.00125         221845.6749    
python main.py no_sleeps .00000000000001 foo 10000  1.77s user 2.95s system 174% cpu 2.708 total
```

Ok! We got some error going on here! That's good. But that's still pretty low, considering we had to execute the threads 
simultaneously 1000 times in a row just to see a tiny bit of error. Another consideration is the slowdown--because concurrency 
bugs are by their nature non-deterministic, you will generally need to run a program multiple times under different circumstances
to uncover them. Thus, if we are going to exacerbate the error to a point where it will be noticeable, we need something that
both introduces more error, and is hopefully faster as well.

I knew at this point that I needed to go back to the drawing board, so I looked at the other idea presented in [3]: putting sleep
statements after every line. This would presumably, drop the GIL, and then also call sleep(0) in the underlying C thread, thus allowing
the OS scheduler to put another thread in. This would hopefully be better because we would only be attempting to drop access to GIL after
each line--no more, no less, rather than at some arbitrary interval. 
Let's try that out in this same scenario!

```python
from time import sleep
...
global x
...
def foo():
    global x
    y = x
    sleep(0)
    x = y+1
...
```

```shell
COMBO                                              ERROR (%)  TIME (ms)      
(0.005, foo)                                       1.2875          337625.254     
python main.py no_sleeps .005 foo 1000  1.01s user 1.52s system 394% cpu 0.640 total
```

Now we are actually getting somewhere--finally in the single digit percent error club!

But now we need a way to do this automatically--the process of selecting the portions of code to flag as incorrect is exactly the point
of concurrency fuzzing. If our process requires manual intervention it makes the process of fuzzing orders of magnitude
slower.

To do this I'm going to leverage my fork [4] (mostly tiny compatibility changes) of the `codetransformer`[5] package. This package allows you to define patterns,
and then match them against bytecode, allowing you to dynamically edit the bytecode stream.
At this point it becomes fairly easy--construct a bytecode payload, insert it into the stream at the appropriate points, execute those functions.

But what are those appropriate points? After every tick! Ticks in the interpreter are atomic. To test this out you can put
in some kind of very simple program like this (where `list` has a very large number of elements)

```python
def example(list):
    return 0 in list
import dis
>>> dis.dis(example)
  2           0 LOAD_CONST               1 (0)
              2 LOAD_FAST                0 (list)
              4 CONTAINS_OP              0
              6 RETURN_VALUE
>>>
```

As you can see the `in` operation is a single bytecode instruction. And while it will take a long time to execute, you will not be able to interrupt it:

```shell
^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C[2]    86736 killed     python -c "print(0 in list(range(2**32))[::-1])"
python -c "print(0 in large_list)"  19.04s user 8.33s system 83% cpu 32.867 total
```

So how are we going to inject a GIL release after every instruction? The answer is quite simple:

```python
class InjectSleep(CodeTransformer):
    def __init__(self, func):
        self.__payload = list(Code.from_pyfunc(func))[:-2]

    @pattern(
        instructions.LOAD_GLOBAL | instructions.LOAD_FAST,
        patterns.matchany[patterns.var],
        instructions.STORE_GLOBAL | instructions.STORE_ATTR,
    )
    def entry(self, *args):
        for instr, lineno in zip(self.code, sorted(self.code.lno_of_instr.values())):
            yield from self.__payload
            yield instr
```

Just yield until the cows come home. 

So now we have something that can inject a function call that will drop the GIL *and* sleep the C thread underlying.
The `@pattern` decorator is basically a way to provide several methods that will each independently match sections of bytecode in the same class. In this case, I am focusing on areas of code where we have some kind of read from and write to global memory
with any number of intervening instructions. Ok, let's run it!

```shell
COMBO                                              ERROR (%)  TIME (ms)      
(0.005, foo, time_module_sleep)                    9.625      0.2342121
```

Ok, we are getting somewhere! But if we're being honest with ourselves, we know that there is some performance we can 
squeeze out of this rock (`time.sleep()`) here. Thus, in a manner very similar to God himself at the rock of Horeb, 
we must write a C-extension. Because of 3 macros that `Python.h` helpfully provides, this is only a few lines of code!

```
PyObject* f(PyObject *self, PyObject *args){
    Py_BEGIN_ALLOW_THREADS
    sleep(0.000001);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
```

The macros should be pretty self explanatory, but they release and acquire the GIL.
In between them I have nested a `sleep()` call, which tells the OS level scheduler to kick us off
and schedule some other thread. This is basically exactly what `time.sleep()` does, but it is a bit more lightweight, with
the standard module's implementation coming in at ~180 lines. This is obviously at the sacrifice of basically *everything*
besides performance. It doesn't even allow you to configure how long it is sleeping! That comes later, don't worry.
Before we compare all these different methods, let's make one small improvement to our C-extension.
Because our function is loaded and ran in a Python thread, it allows us to tune the amount of time
that each thread sleeps as a hyperparameter, *and* it will cause a slightly different interleaving each time because everytime we
call it, it will sleep a random amount of time. For my experiments, I found that the following threshold works best:

```python
PyObject* f2(PyObject *self, PyObject *args){
    Py_BEGIN_ALLOW_THREADS
    sleep(((float)rand()/(float)(RAND_MAX))*0.000001);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
```

Ok, now let's run my testbench program on all of these options and see how they do! Here is my code, broken up into a few chunks,
and explained:

```python
from threading import Thread, Lock
import sys
from time import monotonic_ns, sleep
from functools import cache

from codetransformer import CodeTransformer, instructions, pattern, Code, patterns
import numpy as np
from drop_gil import f, f2
```

Imports.

```python
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
```

Ok this is the fun part, these are the two methods we are using to influence the scheduler.
The first is a context manager to automatically set and unset `switchinterval`, and the second is our bytecode injection class as explained above.

```python
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
```

These are all the functions that wrap the function we want to inject. We add a boolean return value so that `no_sleeps`
can signal a later function not to inject any function (rather than dealing with `Nones`), this return value is later chopped
off the payload so it does not influence the execution of the test.

```python
global_v = 0
global_v_2 = 0
global_lock = Lock()


class BasicClass:
    val = 0
    lock = Lock()


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
```

These are all of the different functions that we are going to test. I have two variants, `foo_*` and `foo_object_*`, to 
demonstrate the two different methods (I think) of accessing global memory in Python. And then for each variant, we have the safe and unsafe versions. The safe versions just being the exact same function, but protected by a lock.

```python
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
```

These two functions are where the real work is being done. We have our function that actually injects the function provided if the injection function returns `True`. And then we have `run_scenario`, which transforms the functions, creates a pool of threads to run those transformed functions, runs them, and waits for them to finish. All while timing it and setting the `switchinterval` using our context manager. 

```python
def run_scenario(n_threads, inject, interval, func, num_iter):
    global global_v, global_v_2, global_lock
    expected = n_threads
    label = f"({interval}, {func.__name__}, {inject.__name__})"
    avg_time = []
    avg_error = []
    for _ in range(num_iter):
        global_v = 0
        global_v_2 = 0
        thread_obj = BasicClass()
        avg_time.append((run_threads(n_threads, func, [thread_obj], inject=inject, interval=interval)) / 10 ** 6)
        avg_error.append(100 * (abs(expected - max(global_v, global_v_2, thread_obj.val))) / expected)
    return label, np.mean(avg_time, axis=0), np.mean(avg_error, axis=0)
```

This just a wrapper around `run_threads` that calls it iteratively and computes the average time and error.

```python
def get_func_from_name(name):
    return getattr(sys.modules[__name__], name)

if __name__ == '__main__':
    n_threads = 8
    num_iter = 10_000
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

print("{:<80} {:<15} {:<15}".format('COMBO', 'ERROR (%)', 'TIME (ns)'))

for row in sorted(zip(labels, errors, times), key=lambda x: x[1]):
    print("{:<80} {:<15} {:<15}".format(*row))
```

This is all of the book keeping work for actually running the program. Selecting different options, dispatching scenarios, etc. Now!

Drumroll please!

Here is the output of my program with `num_iter` set to `100_000`:

| COMBO                                                   | ERROR (%) | TIME (ms) |
| ------------------------------------------------------- | --------- | --------- |
| (2.939e-39, foo_safe, no_sleeps)                        | 0.0       | 198.521   |
| (2.939e-39, foo_object_safe, no_sleeps)                 | 0.0       | 197.599   |
| (5.000e-03, foo, no_sleeps)                             | 0.0       | 195.65    |
| (5.000e-03, foo_object, no_sleeps)                      | 0.0       | 195.801   |
| (5.000e-03, foo_safe, no_sleeps)                        | 0.0       | 196.767   |
| (5.000e-03, foo_object_safe, no_sleeps)                 | 0.0       | 196.966   |
| (2.939e-39, foo_safe, my_extension_sleep)               | 0.0       | 381.381   |
| (2.939e-39, foo_object_safe, my_extension_sleep)        | 0.0       | 425.427   |
| (5.000e-03, foo_safe, my_extension_sleep)               | 0.0       | 302.256   |
| (5.000e-03, foo_object_safe, my_extension_sleep)        | 0.0       | 323.053   |
| (2.939e-39, foo_safe, time_module_sleep)                | 0.0       | 252.483   |
| (2.939e-39, foo_object_safe, time_module_sleep)         | 0.0       | 258.803   |
| (5.000e-03, foo_safe, time_module_sleep)                | 0.0       | 232.351   |
| (5.000e-03, foo_object_safe, time_module_sleep)         | 0.0       | 237.78    |
| (2.939e-39, foo_safe, my_extension_sleep_random)        | 0.0       | 385.584   |
| (2.939e-39, foo_object_safe, my_extension_sleep_random) | 0.0       | 445.896   |
| (5.000e-03, foo_safe, my_extension_sleep_random)        | 0.0       | 302.075   |
| (5.000e-03, foo_object_safe, my_extension_sleep_random) | 0.0       | 322.632   |
| (2.939e-39, foo_object, no_sleeps)                      | 0.001     | 196.572   |
| (2.939e-39, foo, no_sleeps)                             | 0.002     | 198.665   |
| (2.939e-39, foo, time_module_sleep)                     | 4.735     | 230.205   |
| (5.000e-03, foo, time_module_sleep)                     | 5.629     | 219.162   |
| (2.939e-39, foo_object, time_module_sleep)              | 7.121     | 248.685   |
| (5.000e-03, foo_object, time_module_sleep)              | 7.973     | 225.954   |
| (5.000e-03, foo, my_extension_sleep_random)             | 13.259    | 256.747   |
| (5.000e-03, foo, my_extension_sleep)                    | 13.448    | 257.194   |
| (2.939e-39, foo, my_extension_sleep_random)             | 19.84     | 300.974   |
| (2.939e-39, foo, my_extension_sleep)                    | 19.98     | 305.203   |
| (5.000e-03, foo_object, my_extension_sleep)             | 20.945    | 274.378   |
| **(5.000e-03, foo_object, my_extension_sleep_random)**  | 21.171    | 274.311   |
| (2.939e-39, foo_object, my_extension_sleep)             | 31.514    | 367.097   |
| **(2.939e-39, foo_object, my_extension_sleep_random)**  | 34.242    | 373.459   |

As you can see from these results, there are very clearly at least two groups. The safe functions all have an error of zero. This is what we expected--but it's good confirmation that we aren't doing *blatantly* wrong. Now for the unsafe ones, at the very bottom is not injecting anything, just doing `sys.setswitchinterval(1/2**128)`. This, thankfully, is much worse than my solution. However, interestingly enough, if you look at the bottom you can see that a low `switchinterval` value seems to have a synergistic effect with my approach, yielding a 13% increase if you compare the two bolded combinations in the table above. Now, you are probably also noticing the rightmost column, which lists the time per run in milliseconds. This increases nearly two fold comparing our best performing solution (373.459ms) with just cranking up the `switchinterval`(198.665ms). However--because our solution induces so much more error than those approaches (34.242% vs. .002%), it makes it worthwhile. Even if it takes twice as long, if you only have to run it one or two times to expose a bug, it's better than something that takes half as long but you have to run 100 times to expose the same bug. 

### Works Cited

[1] https://docs.python.org/3.8/library/sys.html#sys.setswitchinterval

[2] https://www.dabeaz.com/usenix2009/concurrent/Concurrent.pdf

[3] https://stackoverflow.com/questions/64207879/how-to-detect-data-races-in-python-applications

[4] https://github.com/juliusgeo/codetransformer

[5] https://github.com/llllllllll/codetransformer

**Acknowledgements**

Cool papers about concurrency fuzzing that inspired this article even though I did not directly cite them:

[*] Context-Sensitive and Directional Concurrency Fuzzing for Data-Race Detection
    https://www-users.cse.umn.edu/~kjlu/papers/conzzer.pdf

[*] ConFuzz—A Concurrency Fuzzer
    https://wcventure.github.io/FuzzingPaper/Paper/AISC19_ConFuzz.pdf
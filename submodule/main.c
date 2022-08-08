#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

PyObject* f(PyObject *self, PyObject *args){
    Py_BEGIN_ALLOW_THREADS
    sleep(0.0000000000000000000000000000001);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyMethodDef drop_gilMethods[] = {
    {"f",  f, METH_VARARGS,
     "Drop the GIL"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef drop_gilmodule = {
    PyModuleDef_HEAD_INIT,
    "drop_gil",   /* name of module */
    NULL, /* module documentation, may be NULL */
    0,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    drop_gilMethods,
};
PyMODINIT_FUNC
PyInit_drop_gil(void)
{
    return PyModuleDef_Init(&drop_gilmodule);
}
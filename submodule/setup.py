from setuptools import setup, Extension

module1 = Extension('drop_gil',
                    sources = ['main.c'])

setup (name = 'drop_gil',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])
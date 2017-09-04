#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" setup file for DyBM. """

__author__ = "Hiroshi Kajino"
__version__ = "3.2"
__date__ = "December 22, 2016"
__copyright__ = "(C) Copyright IBM Corp. 2016" 

from setuptools import setup, find_packages
import sys
import os

setup(
    name = "pydybm",
    version = "3.2.1",
    author = "DyBM developers at IBM Research - Tokyo",
    package_dir = {"": "src"},
    packages = find_packages(),
    test_suite = "tests",
)

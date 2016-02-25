#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from debugger.io.dataload import load_mnist


@pytest.fixture
def mnist_sets():
    return load_mnist()

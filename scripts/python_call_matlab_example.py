#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:35:38 2020

@author: brian
"""

import matlab.engine
eng = matlab.engine.start_matlab()
a=eng.engine_python_test()
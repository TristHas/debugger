#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from debugger.conf import CSV_SEP
from conf import SYNC, FAIL

def list_to_csv(input):
    return CSV_SEP.join(map(str,input))

def send_data(soc, mess):
    soc.sendall(mess)
    while True:
        data = soc.recv(8)
        if data == SYNC:
            break
        if data == FAIL:
            break
    return data

def recv_data(soc):
    data = soc.recv(4096)
    soc.sendall(SYNC)
    return data





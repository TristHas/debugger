#!/usr/bin/env python
# -*- coding: utf-8 -*-

###
###     Ctrl Messages
###
STOP_ALL        = '0'
START_RECORD    = '1'
STOP_RECORD     = '2'
START_SEND      = '3'
STOP_SEND       = '4'
START_STORE     = '5'
STOP_STORE      = '6'
START_TRAIN     = '7'
STOP_TRAIN      = '8'
PAUSE_TRAIN     = '9'
RESUME_TRAIN    = '10'
FAIL            = 'fail'
SYNC            = 'sync'
MSG_SEP         = '&&&'

###
###     Net communication
###
IP_1            = "127.0.0.1"
SOC_PORT_CTRL   = 6004
SOC_PORT_DATA   = 6006
LOGIN           = 'd-fr-mac0002'
PWD             = 'Altran2014'

import os

V_SILENT        = 0
V_WARN          = 1
V_INFO          = 2
V_VERBOSE       = 3
V_DEBUG         = 4

DATA_DIR        = '/tmp/debugger/data'
LOG_DIR         = '/tmp/debugger/data/log'
###
###     Directory
###
# Ubuntu /home/tristan/workspace/perf

###
###     Net communication
###
IP_1            = "127.0.0.1"
SOC_PORT_CTRL   = 6004
SOC_PORT_DATA   = 6006
LOGIN           = 'd-fr-mac0002'
PWD             = 'Altran2014'

###
###     Logging
###
V_SILENT                = -1
V_ERROR                 = 0
V_WARN                  = 1
V_INFO                  = 2
V_VERBOSE               = 3
V_DEBUG                 = 4
DATA_LOG_FILE           = os.path.join(LOG_DIR, 'server_data.log')
SERV_LOG_FILE           = os.path.join(LOG_DIR, 'server_ctrl.log')
CLIENT_LOG_FILE         = os.path.join(LOG_DIR, 'client_ctrl.log')
SGD_LOG_FILE            = os.path.join(LOG_DIR, 'main_lgd.log')
PRINT_LOG_FILE          = os.path.join(LOG_DIR, 'client_print.log')
DATA_CLIENT_LOG_FILE    = os.path.join(LOG_DIR, 'client_data.log')

###
###     DEFAULT var
###
D_VERB          = V_DEBUG
D_TIMEOUT       = 10000
D_STEP          = 1

###
###     Store var
###
CSV_SEP         = ','

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

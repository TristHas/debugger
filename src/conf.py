import os

V_SILENT        = 0
V_WARN          = 1
V_INFO          = 2
V_VERBOSE       = 3
V_DEBUG         = 4

LOG_DIR = '/home/tristan/workspace/debugger/data'
LOCAL_DATA_DIR = LOG_DIR
SGD_LOG_FILE = os.path.join(LOG_DIR, 'lgd.log')

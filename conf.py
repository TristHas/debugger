import os

###
###     Directory
###
LOCAL_ROOT_DIR  = '/Users/d-fr-mac0002/Desktop/vision/debugger'
DEPLOY_DIR      = '/tmp/debugger/deploy'
DATA_DIR        = '/tmp/debugger/data'
LOG_DIR         = '/tmp/debugger/data/log'


###
###     Logging
###
V_SILENT                = -1
V_HISTORY               = 0
V_ERROR                 = 1
V_WARN                  = 2
V_INFO                  = 3
V_VERBOSE               = 4
V_DEBUG                 = 5

DATA_LOG_FILE           = os.path.join(LOG_DIR, 'server_data.log')
SERV_LOG_FILE           = os.path.join(LOG_DIR, 'server_ctrl.log')
CLIENT_LOG_FILE         = os.path.join(LOG_DIR, 'client_ctrl.log')
CTRL_LOG_FILE           = os.path.join(LOG_DIR, 'control.log')
CANVAS_LOG_FILE         = os.path.join(LOG_DIR, 'canvas.log')
PROCESSOR_LOG_FILE      = os.path.join(LOG_DIR, 'processor.log')
DATA_CLIENT_LOG_FILE    = os.path.join(LOG_DIR, 'client_data.log')
TRAINER_LOG_FILE        = os.path.join(LOG_DIR, 'trainer.log')
MODEL_LOG_FILE          = os.path.join(LOG_DIR, 'model.log')
UTIL_FILE               = os.path.join(LOG_DIR, 'util')


###
###     DEFAULT var
###
D_VERB          = V_DEBUG
D_TIMEOUT       = 10000
D_STEP          = 1

###
###     Serialization var
###
CSV_SEP         = ','


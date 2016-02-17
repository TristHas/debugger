#!/usr/bin/env python
# -*- coding: utf-8 -*-

from debugger.conf import *
from conf import *
from fabric.api import run, cd, env, put, get
import time, os


def deploy_server(ip = IP_1):
    env.user        = LOGIN
    env.password    = PWD
    env.host_string = ip
    clean_server(ip)
    for item in os.walk(LOCAL_ROOT_DIR):
        if not (item[0] == '.' or '.git' in item[0]):
            directory_path = os.path.join(DEPLOY_DIR, os.path.basename(item[0]))
            print DEPLOY_DIR
            print item[0]
            run("mkdir {}".format(directory_path))
            for file in item[2]:
                file_path = os.path.join(directory_path, item[2])
                put(file_path)

def run_server(ip = IP_1):
    env.user        = LOGIN
    env.password    = PWD
    env.host_string = ip
    run('PYTHONPATH=$PYTHONPATH:{} python {}/debugger/network/record_server.py < /dev/null > /dev/null 2>&1 &'.format(DEPLOY_DIR), pty = False)
    print 'Has run server'
    time.sleep(2)

def kill_server(ip = IP_1):
    env.user        = LOGIN
    env.password    = PWD
    env.host_string = ip
    x = run('ps aux | grep record_server.py')
    y = x.split('\n')
    pid = y[0].split()[1]
    if len(y) == 3:
        run('kill -9 {}'.format(pid))
        print 'has killed {}'.format(y[0].split()[-1])
    else:
        print 'Nothing has been killed'

def clean_server(ip = IP_1):
    env.user        = LOGIN
    env.password    = PWD
    env.host_string = ip

    folders = run("ls {}".format('/tmp'))
    if "debugger" in folders:
        run('rm -r {}'.format('/tmp/debugger'))


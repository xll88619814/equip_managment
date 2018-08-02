#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
sys.path.append("../")
from config import config
from bottle import route, run
import json
from db_connecttion.MySqlConn_gxxj import Mysql


config.DATA_PATH

def get_bean(file_id):
    mysql = Mysql()
    sql = "SELECT file_url FROM ir_file_info WHERE file_id = 'fc57fd0fab5e4fe5bbe3946767fd33f0'"
    result = mysql.getAll(sql)
    im_file = os.path.join(config.DATA_PATH, result)
    return im_file


@route("/:args")
def index(args):
    file_id = args
    print("input args: file_id={}".format(file_id))

    im_file = get_bean(file_id)
    print("image file: {}".format(im_file))


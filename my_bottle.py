#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
#sys.path.append('code')
sys.path.append("..")
sys.path.append('code/train_nummodel')
from config import config
from bottle import route, run
import json
from db_connecttion.MySqlConn_gxxj import Mysql
from code.gettags import detecting


config.DATA_PATH

def get_bean(file_id):
    mysql = Mysql()
    sql = "SELECT file_url FROM ir_file_info WHERE file_id = '{}'".format(file_id)
    result = mysql.getAll(sql)[0]
    im_file = result["file_url"][1:] if result["file_url"][0]=='/' else result["file_url"] 
    im_file = os.path.join(config.DATA_PATH, im_file)
    return im_file


@route("/equip/:args")
def index(args):
    file_id, im_type= args.split(',')
    print("input args: file_id={}".format(file_id))

    im_file = get_bean(file_id)
    print("image file: {}".format(im_file))

    result = detecting(im_file, im_type)
    return json.dumps(result)

run(host='localhost', port=2334)

# if __name__ =="__main__":
#     file_id = 'e066bcd927de409fbaf0648ab95fae82'
#     im_file  = get_bean(file_id)
#     print("image file: {}".format(im_file))
# 
#     ok, result, result_file = detecting(im_file, im_type)
#     print(json.dumps(result))

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
import numpy as np
from db_connecttion.MySqlConn_gxxj import Mysql
from code.locate_u import detecting
from fdfs_client.client import *


config.DATA_PATH

def get_bean(file_id):
    mysql = Mysql()
    sql = "SELECT file_url FROM rb_file_info WHERE file_id = '{}'".format(file_id)
    result = mysql.getAll(sql)[0]
    im_file = result["file_url"][1:] if result["file_url"][0]=='/' else result["file_url"] 
    im_file = os.path.join(config.DATA_PATH, im_file)
    return im_file


def update_bean(file_id, upload_url):
    mysql = Mysql()
    sql = "UPDATE  rb_file_info SET detect_result_url = %s, detect_flag = 1   WHERE file_id = %s "
    param = (upload_url, file_id)
    result = mysql.update(sql, param)
    if result:
        return result
    else:
        return False


def upload_file(file_address):
    print("upload file:")
    print(file_address)
    client = Fdfs_client(config.FDFS_CONFIG)
    result = client.upload_by_filename(file_address)
    time.sleep(2)
    print("fdfs upload")
    print(result)
    upload_url = result['Remote file_id']
    return upload_url





@route("/equip/:args")
def index(args):
    r = args.split(',', 2)
    file_id = r[0]
    angle = r[1]
    detect_setting=r[2]
    print("input args: file_id={}".format(file_id))
    print("input args: detect_setting={}".format(detect_setting))
    im_file = get_bean(file_id)
    print("image file: {}".format(im_file))
    print("input args: angle={}".format(angle))
    
    ok, details,result_file, result_u,light_ok,light_u = detecting(im_file,angle,detect_setting)
    final_result = {}
    final_details =[]
    print("details is{} ".format(details))
   # for d in details:
   #     d["U"] = ",".join(map(str, d["U"]))
   #     final_details.append(d)
    
    print(" done ")
   
    
    final_result["result"]=str(ok)
    final_result["details"] = details
    final_result["result_u"] = result_u

    final_result["light_ok"] = str(light_ok)
    final_result["light_u"] = light_u

    print("result_u : {}".format(result_u))
    detect_url = upload_file(result_file)
    print("detect_url : {}".format(detect_url))
    update_result = update_bean(file_id,detect_url)
    print("update_result : {}".format(update_result))
    print("finale result :{}".format(final_result))    
    return json.dumps(final_result)


run(host='218.85.116.194', port=2445)

# if __name__ =="__main__":
#     file_id = 'e066bcd927de409fbaf0648ab95fae82'
#     im_file  = get_bean(file_id)
#     print("image file: {}".format(im_file))
# 
#     ok, result, result_file = detecting(im_file, im_type)
#     print(json.dumps(result))

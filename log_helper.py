import sys
import os
import datetime
import logging
import inspect

logging.basicConfig(filename='info.log', filemode='w', level=logging.INFO)
# 定义要过滤的文件列表
filtered_files = ['transformer.py', 'NormalCell.py', 'WindowAttention.py',
                  'ms_deform_attn_func.py']

def log_print(*args, **kwargs):
    filename = os.path.basename(sys._getframe(1).f_code.co_filename)
    # 检查该文件是否在过滤列表中
    if filename in filtered_files:
        return  # 如果在过滤列表中，则不记录日志，直接返回
    
    time = datetime.datetime.now().strftime('%F_%T.%f')[:-3]
    current_frame = inspect.currentframe()
    call_frame = inspect.getouterframes(current_frame, 2)[1]
    funcname = call_frame.function
    loc = call_frame.lineno

    arg_values = args
    for i, message in enumerate(arg_values, start=1):
        tag = f"ARG{i}"  # Default tag for unnamed args
        log_and_print(time, filename, funcname, loc, tag, message)
    for tag, message in kwargs.items():
        log_and_print(time, filename, funcname, loc, tag, message)

def log_and_print(time, filename, funcname, loc, tag, message):
    log_message = f'[{time}]: {funcname}: [{filename}: {loc}]: {tag} = {message}'
    print(log_message)
    logging.info(log_message)
#!/usr/bin/python3 -u

import ctf

def print_title(title_name,sep_symbol):
    world=ctf.comm()
    if world.rank()  == 0:
        print(sep_symbol*80)
        num_title_name = len(title_name)
        num_space = int((80-num_title_name)/2)
        print(" "*num_space+title_name+" "*num_space)
        print(sep_symbol*80)
    return

def print_logging_info(*args,**kwargs):
    world = ctf.comm()
    level = 0
    if "level" in kwargs:
        level = kwargs["level"]
    if world.rank() == 0:
        print("    "*level+''.join(str(i) for i in args))
    return

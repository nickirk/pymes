#!/usr/bin/python3 -u

import ctf

def print_title(title_name,sep_symbol,level=1):
    world=ctf.comm()
    if world.rank()  == 0:
        if level == 0:
            level = 1
        length_banner = int(80/level)
        num_title_name = len(title_name)
        if length_banner < num_title_name:
            length_banner = num_title_name+2
        right_shift = int((80-length_banner)/2)
        print(" "*right_shift+sep_symbol*length_banner)
        num_space = int((length_banner-num_title_name)/2)
        print(" "*(right_shift+num_space)+title_name+" "*num_space)
        print(" "*right_shift+sep_symbol*length_banner)
    return

def print_logging_info(*args,**kwargs):
    world = ctf.comm()
    level = 0
    if "level" in kwargs:
        level = kwargs["level"]
    if world.rank() == 0:
        print("    "*level+''.join(str(i) for i in args))
    return

#!/usr/bin/python3

import re


def getvalue(lines, target, typ, required=False, default=None):
    """Get the value of a keyword with a single argument"""
    for line in lines:
        if re.match(target, line) is not None:
            return typ(re.split(r'\s+', line.strip())[-1])
    if required:
        errstr = "Keyword '"+target+"' was not found"
        raise ValueError(errstr)
    elif default is not None:
        return default


def getlist(lines, target, typ, required=False, default=None):
    """Get the value of a keyword with a list of arguments"""
    for line in lines:
        if re.match(target, line) is not None:
            return [typ(x) for x in re.split(r'\s+', line.strip())[1:]]
    if required:
        errstr = "Keyword '"+target+"' was not found"
        raise ValueError(errstr)
    elif default is not None:
        return default
    return []


def getbool(lines, target, required=False, default=None):
    """Get the value for a boolean keyword"""
    for line in lines:
        if re.match(target, line) is not None:
            value = str(re.split(r'\s+', line.strip())[-1])
            if value in ["1","True","true"]:
                return True
            elif value in ["0","False","false"]:
                return False
            else:
                errstr = "Boolean '"+target+"' keyword value '"+value+"' is not valid" 
                raise ValueError(errstr)
    if required:
        errstr = "Keyword '"+target+"' was not found"
        raise ValueError(errstr)
    elif default is not None:
        return default

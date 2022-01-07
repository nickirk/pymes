import sys
import os
import pkg_resources
from pprint import pprint


pprint({
    'sys.version_info': sys.version_info,
    'sys.prefix': sys.prefix,
    'sys.path': sys.path,
    'pkg_resources.working_set': list(pkg_resources.working_set),
    'PATH': os.environ['PATH'].split(os.pathsep),
})
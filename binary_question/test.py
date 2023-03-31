import sys
#  path[0] is reserved for script path 
#needed to acces global files
sys.path.insert(1, os.getcwd())

import os
#needed to acces folder files
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


print(os.getcwd())
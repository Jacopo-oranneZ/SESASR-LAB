import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/l0dz/SESASR-LAB/install/lab02_pkg'

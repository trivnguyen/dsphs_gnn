
import shutil
from setuptools import setup

shutil.copy("config.ini", "src/dsphs_gnn/config.ini")
setup()

import subprocess
import os

import pandas as pd
import numpy as np

from datetime import datetime

import requests
import json

def get_john_hopkins():
    git_pull = subprocess.Popen("/usr/bin/git pull",
                         cwd = os.path.dirname('data/raw/COVID-19_New/COVID-19/'),
                         shell = True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)
    (out,error) = git_pull.communicate()

    print("Error : " + str(error))
    print("out : " + str(out))

if __name__ == '__main__':
    get_john_hopkins()

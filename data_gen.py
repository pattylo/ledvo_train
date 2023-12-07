#!/usr/bin/env python3

import pandas as pd
import train.gendata as ld
import os
   

if __name__ == "__main__":
    print('\033c')
    print("START TO GEN SCRIPT!\n")    
    data_config = ld.gendata(os.getcwd() + "/config/train.yaml")

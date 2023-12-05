import pandas as pd
import train.gendata as ld
import os
   

if __name__ == "__main__":
    print("START TO LOAD DATA!")    
    data_config = ld.gendata(os.getcwd() + "/config/train.yaml")

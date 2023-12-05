
import train.traindata as td
import os

if __name__ == "__main__":
    print("START TO LOAD DATA!")    
    data_config = td.traindata(os.getcwd() + "/config/train.yaml")

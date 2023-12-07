import pandas as pd
import os
import torch
import yaml
import torch
import yaml
from tqdm import tqdm
import multiprocessing
import psutil
import sys
import time
import queue

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split

import torch.optim as optim

import shutil

import train.tcn as tcn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor

import torch.optim as optim
from datetime import datetime


class CustomDataset(Dataset):
    def __init__(self, data_ptsss):
        self.data_ptsss = data_ptsss
        self.num_files = len(self.data_ptsss)
        
        # Load all data into memory
        self.data = self._load_data()

    def _load_data(self):
        all_data = {'inputs': [], 'labels': []}
        # for file_name in self.data_ptsss:
        for file_name in tqdm(
                self.data_ptsss,
                desc="LOAD >>> ",
                unit='iteration',
                total=len(self.data_ptsss)
            ):

            # print(file_name)
            
            data = torch.load(file_name)

            input_tensor = data['inputs']  # Replace 'input' with the actual key in your .pt file
            label_tensor = data['labels']  # Replace 'label' with the actual key in your .pt file
            

            all_data['inputs'].extend(input_tensor)
            all_data['labels'].extend(label_tensor)
            
            ram_info = psutil.virtual_memory()
            if ram_info.used / (1024 ** 3) > 60:
                print('RAM OVERLOAD!')
                sys.exit()

        return all_data

    def __len__(self):
        return len(self.data['inputs'])

    def __getitem__(self, idx):
        return {'inputs': self.data['inputs'][idx], 'labels': self.data['labels'][idx]}

class traindata:
    def __init__(self, config_filepath) -> None:
        yaml_file_path = config_filepath
        
        with open(yaml_file_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)
        
        ####################################################
                        
        self.raw_data_folder = self.yaml_data.get('raw_data_file')
        self.proc_data_folder = self.yaml_data.get('proc_data_file')
        self.min_window_size = self.yaml_data.get('min_window_size')
        self.max_window_size = self.yaml_data.get('max_window_size')
        
        self.epochs = self.yaml_data.get('epochs')
        self.batch_size = int(self.yaml_data.get('batch_size'))
        
        self.input_size = self.yaml_data.get('input_size')
        self.output_size = self.yaml_data.get('output_size')
        
        self.learning_rate = float(self.yaml_data.get('learning_rate'))
        
        self.save_model_file = self.yaml_data.get('save_model_file')
        
        ####################################################
        
        self.epoch_no_save_pt = {key: None for key in range(10, 201, 10)}
            
        self.data_config_load()
        self.data_main_load()
        self.do_train()
        
        pass        
    
    def data_main_load(self):
        custom_dataset = CustomDataset(self.data_of_tensors_pt)
        # Determine the size of the training set (e.g., 80% of the data)
        train_size = int(0.8 * len(custom_dataset))
        test_size = len(custom_dataset) - train_size

        # Use random_split to create training and test datasets
        train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

        # Create DataLoader instances for training and testing
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.all_data_point_no = int(len(custom_dataset))
        self.all_train_batches_no = int(len(self.train_loader))
        self.all_test_batches_no = int(len(self.test_loader))
        
        print(f'ALL DATA POINT NO: {self.all_data_point_no}')
        print(f'ALL TRAIN BATCHES POINT NO: {self.all_train_batches_no}')
        print(f'ALL TRAIN BATCHES POINT NO: {self.all_test_batches_no}')
        print(f'BATCH SIZE: {self.batch_size}')
        
    def data_config_load(self):        
        if os.path.exists(self.proc_data_folder) and os.path.isdir(self.proc_data_folder):
            # Get a list of all subdirectories
            self.data_folders = [name for name in os.listdir(self.proc_data_folder) if os.path.isdir(os.path.join(self.proc_data_folder, name))]
        else:
            print(f"The specified path '{self.proc_data_folder}' is not a directory or does not exist.")

        print(self.data_folders)
        
        window_sizes = list(range(self.min_window_size, self.max_window_size))    
        self.data_of_tensors_pt = []
    
        for foldername in self.data_folders:
            for window_size in window_sizes:  
                data_loadname = self.proc_data_folder + foldername + '/data_' + foldername + '_' + str(window_size) + '.pt'
                if os.path.exists(data_loadname):
                    self.data_of_tensors_pt.append(data_loadname)
                                        
        print(f'PT FILES SIZE: {len(self.data_of_tensors_pt)}')
    
    
##############################################################################################################################        

    def do_train(self):
        print("NOW START TO TRAIN!")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.network = tcn.LedvoTcn(
            self.input_size,
            self.output_size,
            [64, 64, 64, 64, 128, 128, 128],
            kernel_size=2,
            dropout=0.2,
            activation="GELU"
        )
        
        print('\n\n\nMODEL AND DATA ALL LOADED!\n\n\n')
        
        self.network.to(self.device)
        self.optimizer_init()
        
        logname = 'log.txt'
        self.f = open(logname, 'w')
        self.f.write('Run Start Time: ' + str(time.ctime()))
        self.f.write('\tLearning Rate\t%f\n' % self.learning_rate)
        
        self.f.write('Data Point No.:\t%f\n' % int(self.all_data_point_no))
        self.f.write('Batch Size:\t%f\n' % int(self.batch_size))
        self.f.write('Train Batch No.:\t%f\n' % int(self.all_train_batches_no))
        self.f.write('Test Batch No.:\t%f\n' % int(self.all_test_batches_no))
        
        self.start_time = datetime.now()
        
        best_loss = float('inf')
        
        for epoch in tqdm(
            range(self.epochs),
            desc="TRAIN >>>",
            unit='epoch',
            total=self.epochs
        ):
            loss_temp = self.one_epoch_train(epoch)  

            if (epoch + 1) in self.epoch_no_save_pt:
                model_pt_name = self.save_model_file + '_' + str(epoch + 1) + '.pt'
                if os.path.exists(model_pt_name):
                    os.remove(model_pt_name)
                torch.save(self.network.state_dict(), model_pt_name)
                
                if loss_temp < best_loss:
                    model_pt_name = self.save_model_file + '_best.pt'
                    
                    if os.path.exists(model_pt_name):
                        os.remove(model_pt_name)
                    torch.save(self.network.state_dict(), model_pt_name)
                    
                    best_loss = loss_temp
        
        model_pt_name = self.save_model_file + '_final.pt'
        if os.path.exists(model_pt_name):
            os.remove(model_pt_name)
        torch.save(self.network.state_dict(), model_pt_name)
        
                
    def one_epoch_train(self, epoch):
        
        epoch_start_time = datetime.now()
        
        # train
        epoch_loss = 0.0
        train_iteration = 0
        for i, data in enumerate(self.train_loader):

            input_train_data = data["inputs"]
            input_train_target = data["labels"]

            input_train_data = input_train_data.to(self.device) 
            input_train_target = input_train_target.to(self.device) 
            
            self.optimizer.zero_grad()
            
            outputs = self.network.forward(input_train_data)
            batch_loss = self.loss(dp_preds=outputs, dp_targets=input_train_target)
                        
            batch_loss = batch_loss.mean()
            batch_loss.backward()
            
            self.optimizer.step()
            
            epoch_loss += batch_loss.item()
            
            train_iteration = train_iteration + 1

        # test
        test_loss_all = 0
        test_iteration = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                
                input_test_data = data["inputs"]
                input_test_target = data["labels"]

                input_test_data = input_test_data.to(self.device) 
                input_test_target = input_test_target.to(self.device) 
                
                outputs = self.network.forward(input_test_data)
                test_loss = self.loss(dp_preds=outputs, dp_targets=input_test_target)
                test_loss = test_loss.mean()
                test_loss_all = test_loss_all + test_loss
                test_iteration = test_iteration + 1
                

        # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        self.f.write("Epoch\t%d\tLoss\t%f\t%f\t,\tTest\tLoss\t%f\t%f\n" % (epoch + 1, epoch_loss / train_iteration, epoch_loss, test_loss_all / test_iteration, test_loss_all))
        
        
        epoch_end_time = datetime.now()
                
        print(f'Epoch:{epoch + 1} Complete!')
        print(f'Loss:{epoch_loss}...\n')

        print('This Epoch End Time: ' + str(time.ctime()))
        print()
        elapsed_time = epoch_end_time - epoch_start_time
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Epoch Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")
        
        elapsed_time = epoch_end_time - self.start_time
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")
        print("\n========================================================")
        print("========================================================\n\n")
        
        return epoch_loss
        
    def loss(self, dp_preds, dp_targets):
        errs = dp_preds - dp_targets
        loss = torch.mean((errs).pow(2))
        
        return loss

    
    def optimizer_init(self):
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                factor=0.1, 
                patience=10, 
                verbose=True, 
                eps=1e-12
            )

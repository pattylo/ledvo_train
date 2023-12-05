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
import shutil

class gendata:
    def __init__(self, config_filepath) -> None:
        yaml_file_path = config_filepath
        
        with open(yaml_file_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)
                
        self.raw_data_folder = self.yaml_data.get('raw_data_file')
        self.proc_data_folder = self.yaml_data.get('proc_data_file')
        self.min_window_size = self.yaml_data.get('min_window_size')
        self.max_window_size = self.yaml_data.get('max_window_size')
        
        self.delete_previous = self.yaml_data.get('delete_previous')


        # Check if the specified path is a directory
        if os.path.exists(self.proc_data_folder) and os.path.isdir(self.proc_data_folder):
            # Get a list of all subdirectories
            subdirectories = [name for name in os.listdir(self.proc_data_folder) if os.path.isdir(os.path.join(self.proc_data_folder, name))]

            # Loop through each subdirectory and delete it
            for subdir in subdirectories:
                folder_path = os.path.join(self.proc_data_folder, subdir)
                try:
                    shutil.rmtree(folder_path)
                    print(f"Directory '{folder_path}' removed successfully.")
                except OSError as e:
                    print(f"Error: {e}")
        else:
            print(f"The specified path '{self.proc_data_folder}' is not a directory or does not exist.")

        print()

        self.generate_data = self.yaml_data.get('generate_data')
        if self.generate_data:
            self.gen_all_data()   
    
    def gen_all_data(self):
        begin_time = time.time()
        self.load_all()
        end_time = time.time()
        run_time = (end_time - begin_time)
        print('Runtime:\t%f\n' % run_time)

    def load_all(self):
        self.file_names = [file for file in os.listdir(self.raw_data_folder) if os.path.isfile(os.path.join(self.raw_data_folder, file))]
        print(f'RAW DATA FILENAMES:\n{self.file_names}\n')
        
        processes = []
        
        # Create and start each worker process
        num_processes = len(self.file_names)
        
        for i in range(num_processes):
            process = multiprocessing.Process(
                target=self.main_loading, 
                args=(
                    self.raw_data_folder,
                    self.file_names[i],
                    self.min_window_size,
                    self.max_window_size,
                    i
                )
            )
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()            
            
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nLOAD DATA DONE!')
        
                
    def main_loading(self, folderpath, filename, min_window_size, max_window_size, index):
        # print(folderpath + filename)
        window_sizes = list(range(min_window_size, max_window_size))
        data_raw = pd.read_csv(folderpath + filename)
        # print(data_raw.shape)
        
        data_input_per_chunk = []
        data_label_per_chunk = []
        
        for window_size in tqdm(
            window_sizes, 
            desc=filename, 
            unit='iteration', 
            position=index,
            total=len(window_sizes)
        ):
            for index, what in enumerate(data_raw['t']): # data raw will be the whole block        
                                        
                if index+window_size == len(data_raw['t']):
                    break
                
                #input
                input_selected_data = data_raw.iloc[index:index + window_size, 1:5]
                input_tensor_data = torch.tensor(input_selected_data.values, dtype=torch.float32)
                input_tensor_data = input_tensor_data.t()
                
                padding = (
                    max_window_size - input_tensor_data.size(1), 0,  # Padding for the second dimension
                    4 - input_tensor_data.size(0), 0   # Padding for the first dimension
                )
                input_tensor_data = torch.nn.functional.pad(
                    input=input_tensor_data,
                    pad=padding,
                    value=0.0
                )
                
                data_input_per_chunk.append(input_tensor_data)
                
                # target
                target_selected_data = data_raw.iloc[index+window_size,8:11] - data_raw.iloc[index,8:11]
                target_tensor_data = torch.tensor(target_selected_data.values, dtype=torch.float32)
                target_tensor_data = target_tensor_data.t()
                
                data_label_per_chunk.append(target_tensor_data)
            
            
            directory_path = self.proc_data_folder + filename.rstrip('.csv') + '/'

            # Check if the directory already exists
            if not os.path.exists(directory_path):
                # Create the directory
                os.makedirs(directory_path)
            else:
                pass

            data_dict = {'inputs': data_input_per_chunk, 'labels': data_label_per_chunk}
            file_name_pt = directory_path + 'data_' + filename.rstrip('.csv') + '_' + str(window_size) + '.pt'
            # label_file_name_pt = directory_path + 'label_' + filename.rstrip('.csv') + '_' + str(window_size) + '.pt'
            # torch.save(d)
            
            # print(input_file_name_pt)
            if os.path.exists(file_name_pt):
                os.remove(file_name_pt)
            torch.save(data_dict, file_name_pt)
            data_input_per_chunk.clear()
            data_label_per_chunk.clear()
            
            memory_info = psutil.virtual_memory()
            
            if memory_info.used / (1024 ** 3) > 60:
                print("MEMORY OVERLOAD!")               
                sys.exit()
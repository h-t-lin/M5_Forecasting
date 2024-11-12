import os
import numpy as np
import torch
from torch.utils.data import Dataset


class M5_Dataset(Dataset):
    def __init__(self, data_path:str, Twin=[7*8,7*13], Tpred=7, is_inference=False, mode='full', num_sampling=3200):
        '''mode = "full" | "random"'''
        self.data_path = data_path
        self.store_dir = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
        self.calendar_dir = 'DInfo'
        self.num_Store = len(self.store_dir)
        self.num_Item = 3049

        self.val_date = (1913,1941)  # starting and endding date of validation seq
        self.Tmin = Twin[0]  # minimal seq length
        self.Tmax = Twin[1]  # maximal seq length
        self.Tpred = Tpred  # predicted seq length
        self.time_range = self.val_date[0] - self.Tmax - self.Tpred  # latest starting date of input seq

        self.mode = mode
        self.num_sampling = num_sampling
        if mode=='full': self.len_data = self.num_Store * self.time_range
        elif mode=='random': self.len_data = int(num_sampling//self.num_Store*self.num_Store)
        else: raise ValueError(f'Mode has to be either "full" or "random".')

        self.is_inference = is_inference
        # if is_inference: self.len_data = int(self.num_Store * np.ceil((self.val_date[1]-self.val_date[0])/Tpred))
        if is_inference: self.len_data = self.num_Store

    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        seq_size = self.Tmin
        # seq_size = np.random.randint(self.Tmin, self.Tmax+1)

        store, date = self.id_to_store_date(idx)
        if self.mode=='random':
            date = self.random_sampling()
        if self.is_inference:
            date = self.val_date[0] - seq_size

        Selling_data = self.get_files(os.path.join(self.data_path, self.store_dir[store]), date, seq_size+self.Tpred)  # shape: [seq_size + Tpred, num_Item]
        gt = Selling_data[-self.Tpred:]  # shape: [Tpred, num_Item]
        Selling_data = Selling_data[:-self.Tpred]  # shape: [seq_size, num_Item]

        Date_data = self.get_files(os.path.join(self.data_path, self.calendar_dir), date, seq_size+self.Tpred)  # shape: [seq_size + Tpred, num_Item]

        Store_data = np.zeros(self.num_Store, dtype=int)  # shape: [num_Store]
        Store_data[store] = 1

        # Price_data
        
        gt, Selling_data, Date_data, Store_data = \
            torch.tensor(gt, dtype=torch.float32), torch.tensor(Selling_data, dtype=torch.float32), torch.tensor(Date_data, dtype=torch.float32), torch.tensor(Store_data, dtype=torch.float32)

        return gt, Selling_data, Date_data, Store_data
    
    def random_sampling(self):
        pass

    def id_to_store_date(self, idx):
        num_item_per_store = self.len_data // self.num_Store
        store_id = idx // num_item_per_store
        date = idx % num_item_per_store
        return store_id, date

    def get_files(self, dir:str, start:int, total:int):
        flist = [os.path.join(dir, f"d_{date}.npy") for date in range(start, start+total)]
        data = []
        for fname in flist:
            data.append(np.load(fname))
        return np.array(data)

import os
import pandas
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

CSVfiles = {
    "C": "calendar",
    "SE": "sales_train_evaluation",
    "SV": "sales_train_validation",
    "SM": "sample_submission",
    "SP": "sell_prices",
}
FILE = 'SE'
N_DAYS = 1941
N_ITEMS = 3049
STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
DEPT = {'HOBBIES_1':416, 'HOBBIES_2':149, 'HOUSEHOLD_1':532, 'HOUSEHOLD_2':515, 'FOODS_1':216, 'FOODS_2':398, 'FOODS_3':823,}
dataset_path = "C:\\Users\\htlin\\Desktop\\SideProj\\M5Forcasting\\m5-dataset_v1"
file_dir = "C:\\Users\\htlin\\Desktop\\SideProj\\M5Forcasting\\m5-forecasting-accuracy"


def main0():
    # Read .csv file
    df_csv = pandas.read_csv(os.path.join(file_dir, f"{CSVfiles['SV']}.csv"))

    # store_data: [store][day, item]
    store_data = {k:None for k in STORES}
    # for idx, store in enumerate(store_data):
    #     temp_list = []
    #     for key in df_csv:
    #         if not key.startswith('d_'): continue
    #         temp_list.append(np.asarray(df_csv[key][N_ITEMS*idx:N_ITEMS*(idx+1)], dtype=np.int32))
    #     store_data[store] = np.array(temp_list)

    df_csv = pandas.read_csv(os.path.join(file_dir, f"{CSVfiles['SP']}.csv"))

    # price_data: [store][day, item]
    price_data = {k:{} for k in STORES}
    for store, item, wk in zip(df_csv['store_id'], df_csv['item_id'], df_csv['wm_yr_wk']):
        try:
            price_data[store][item].append(wk)
        except:
            price_data[store][item] = []
            price_data[store][item].append(wk)

    with pandas.ExcelWriter(os.path.join(file_dir, f"corr.xlsx")) as wr:
        startid = 0
        for dept, n_dept in DEPT.items():
            MA = (1,7,14)
            meanR = {k:np.zeros([len(STORES), len(STORES)], dtype=np.float32) for k in MA}
            for idx in tqdm(range(n_dept)):
                item_data = np.array([dt[:,idx+startid] for dt in store_data.values()], dtype=np.int32)
                R = np.corrcoef(item_data)
                meanR[1] = meanR[1] + R

                # moving average
                data_ma = np.zeros_like(item_data, dtype=np.float32)
                for ma in MA[1:]:
                    for idx in range(len(STORES)):
                        data_ma[idx] = moving_average(item_data[idx], win_size=ma)
                    R = np.corrcoef(data_ma)
                    meanR[ma] = meanR[ma] + R

            for ma, mR in meanR.items():
                mR = mR/n_dept
                df_excel = pandas.DataFrame(mR)
                df_excel.to_excel(wr, sheet_name=f'{dept}-{ma}ma')
            
            startid+=n_dept

def FREQ_of_Items():
    # Read .csv file
    df_csv = pandas.read_csv(os.path.join(file_dir, f"{CSVfiles[FILE]}.csv"))

    # store_data: [store][day, item]
    store_data = {k:None for k in STORES}
    for idx, store in enumerate(store_data):
        temp_list = []
        for key in df_csv:
            if not key.startswith('d_'): continue
            temp_list.append(np.asarray(df_csv[key][N_ITEMS*idx:N_ITEMS*(idx+1)], dtype=np.int32))
        store_data[store] = np.array(temp_list)

    for store in store_data:
        for idx in np.random.randint(0, N_ITEMS-1500, size=100):
            item_data = store_data[store][:,idx]
            fq_data = np.fft.fft(item_data)
            axis = np.fft.fftfreq(item_data.shape[-1])

            # 只顯示正頻率部分
            pos_mask = axis >= 0
            fq_data = fq_data[pos_mask]
            axis = axis[pos_mask]
            
            # 繪製頻譜圖
            plt.figure(figsize=(15, 9))
            plt.plot(axis, np.abs(fq_data), color='blue')
            for T, c in zip((7,14,30, 90, 7/2, 7/3), ('orange', 'gold', 'pink', 'violet', 'tan', 'tan')):
                plt.axvline(x=1/T, color=c, linestyle='--', alpha=0.6)
                plt.text(1/T, plt.ylim()[1]*0.9, f"T={np.round(T,2)}d", color=c, alpha=0.6, ha='center', fontsize=8)
            plt.title(f"Frequency Spectrum - {df_csv['item_id'][idx]}")
            plt.xlabel("Frequency")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

def create_dataset_date():
    # Read .csv file
    df_csv = pandas.read_csv(os.path.join(file_dir, f"{CSVfiles['C']}.csv"))

    # EventsT = {}
    # for event in df_csv['event_type_1']:
    #     try: EventsT[event]+=1
    #     except: EventsT[event] = 0
    # for event in df_csv['event_type_2']:
    #     try: EventsT[event]+=1
    #     except: EventsT[event] = 0

    # Events = {}
    # for event in df_csv['event_name_1']:
    #     try: Events[event]+=1
    #     except: Events[event] = 0
    # for event in df_csv['event_name_2']:
    #     try: Events[event]+=1
    #     except: Events[event] = 0    

    DateInfo = {}
    DateInfo['date'] = np.array([np.array([yr, mh, wk, wid], dtype=int) for yr, mh, wk, wid in zip(df_csv['year'], df_csv['month'], df_csv['wday'], df_csv['wm_yr_wk'])], dtype=int)
    DateInfo['snap'] = np.array([np.array([ca, tx, wi], dtype=int) for ca, tx, wi in zip(df_csv['snap_CA'], df_csv['snap_TX'],df_csv['snap_WI'])], dtype=int)
    Events = []
    event_types = {'Sporting': 0, 'Cultural': 1, 'National': 2, 'Religious': 3}
    for ev1, ev2 in zip(df_csv['event_type_1'], df_csv['event_type_2']):
        temp = np.array([0,0,0,0], dtype=int)
        try:
            temp[event_types[ev1]] += 1
        except: pass
        try:
            temp[event_types[ev2]] += 1
        except: pass
        Events.append(temp)
    DateInfo['event'] = np.array(Events, dtype=int)

    os.makedirs(dataset_path, exist_ok=True)
    subdir = os.path.join(dataset_path, 'DInfo')
    os.makedirs(subdir, exist_ok=True)
    for day in range(DateInfo['event'].shape[0]):
        fname = f"d_{day}.npy"
        info = np.concatenate((DateInfo['date'][day,:], DateInfo['event'][day,:], DateInfo['snap'][day,:]), dtype=int)
        np.save(os.path.join(subdir, fname), info)

def create_dataset_selling():
    # Read .csv file
    df_csv = pandas.read_csv(os.path.join(file_dir, f"{CSVfiles['SE']}.csv"))

    # store_data: [store][day, item]
    store_data = {k:None for k in STORES}
    for idx, store in enumerate(store_data):
        temp_list = []
        for key in df_csv:
            if not key.startswith('d_'): continue
            temp_list.append(np.asarray(df_csv[key][N_ITEMS*idx:N_ITEMS*(idx+1)], dtype=np.int32))
        store_data[store] = np.array(temp_list)
    
    os.makedirs(dataset_path, exist_ok=True)
    for idx, store in enumerate(store_data):
        subdir = os.path.join(dataset_path, store)
        os.makedirs(subdir, exist_ok=True)

        for day in range(N_DAYS):
            fname = f"d_{day}.npy"
            np.save(os.path.join(subdir, fname), store_data[store][day, :])
    
def CORR_bt_Stores():
    # Read .csv file
    df_csv = pandas.read_csv(os.path.join(file_dir, f"{CSVfiles[FILE]}.csv"))

    # store_data: [store][day, item]
    store_data = {k:None for k in STORES}
    for idx, store in enumerate(store_data):
        temp_list = []
        for key in df_csv:
            if not key.startswith('d_'): continue
            temp_list.append(np.asarray(df_csv[key][N_ITEMS*idx:N_ITEMS*(idx+1)], dtype=np.int32))
        store_data[store] = np.array(temp_list)

    with pandas.ExcelWriter(os.path.join(file_dir, f"corr.xlsx")) as wr:
        startid = 0
        for dept, n_dept in DEPT.items():
            MA = (1,7,14)
            meanR = {k:np.zeros([len(STORES), len(STORES)], dtype=np.float32) for k in MA}
            for idx in tqdm(range(n_dept)):
                item_data = np.array([dt[:,idx+startid] for dt in store_data.values()], dtype=np.int32)
                R = np.corrcoef(item_data)
                meanR[1] = meanR[1] + R

                # moving average
                data_ma = np.zeros_like(item_data, dtype=np.float32)
                for ma in MA[1:]:
                    for idx in range(len(STORES)):
                        data_ma[idx] = moving_average(item_data[idx], win_size=ma)
                    R = np.corrcoef(data_ma)
                    meanR[ma] = meanR[ma] + R

            for ma, mR in meanR.items():
                mR = mR/n_dept
                df_excel = pandas.DataFrame(mR)
                df_excel.to_excel(wr, sheet_name=f'{dept}-{ma}ma')
            
            startid+=n_dept

def moving_average(data: np.ndarray, win_size=3):
    L = len(data)
    dataMA = np.zeros_like(data, dtype=np.float32)
    for idx in range(0, win_size):
        dataMA[idx] = np.mean(data[:idx+1])
    for idx in range(win_size, L):
        dataMA[idx] = np.mean(data[idx+1-win_size:idx+1])
    return dataMA


if __name__ == '__main__':
    # main()
    create_dataset_selling()
    # create_dataset_date()
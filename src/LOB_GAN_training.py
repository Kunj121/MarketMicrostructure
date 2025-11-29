# -*- coding: utf-8 -*-
"""
Created on Wed October 8 09:40:23 2025

@author: hongs; adapted from the orignal copy by Mr. GUAN Chenjiong, 2025.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from settings import G_PARAMS, D_PARAMS

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / "assignment4_datafiles"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output_train'

PARAMS_DIR = None

def log_params(params_idx: int, lr_g: float, lr_d: float):
    """
    Append or create logs/params_dict.csv with:
    params, lr_g, lr_d

    Only writes a row if this params_idx does NOT already exist.
    """
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    csv_path = logs_dir / "params_dict.csv"

    # If file exists, check whether this params_idx is already logged
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # if params column exists and this index already logged, do nothing
        if "params" in df.columns and (df["params"] == params_idx).any():
            return  # already logged, nothing to do

        # otherwise, append without header
        new_row = pd.DataFrame(
            {"params": [params_idx], "lr_g": [lr_g], "lr_d": [lr_d]}
        )
        new_row.to_csv(csv_path, mode="a", header=False, index=False)
    

def _get_params_output_dir() -> Path:
    """
    Create (once per run) a subfolder under OUTPUT_DIR named
    PARAMS_1, PARAMS_2, ... depending on what already exists.
    """
    global PARAMS_DIR
    if PARAMS_DIR is not None:
        return PARAMS_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find existing PARAMS_* directories
    existing = []
    for p in OUTPUT_DIR.iterdir():
        if p.is_dir() and p.name.startswith("PARAMS_"):
            parts = p.name.split("_")
            if len(parts) == 2:
                try:
                    idx = int(parts[1])
                    existing.append(idx)
                except ValueError:
                    continue

    next_idx = (max(existing) + 1) if existing else 1
    PARAMS_DIR = OUTPUT_DIR / f"PARAMS_{next_idx}"
    PARAMS_DIR.mkdir(exist_ok=False)

    log_params(next_idx, G_PARAMS['lr'], D_PARAMS['lr'])

    return PARAMS_DIR



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.lay2 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay5 = nn.GRU(40, 19, num_layers=1, batch_first=True) # note: one layer should have less than 20 nodes to avoid data repeatation
        self.lay6 = nn.Sequential(nn.Linear(19, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay7 = nn.GRU(40, 20, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 20))
    
    #forward propagation
    def forward(self, x):
        y, _ = self.lay1(x)
        z = self.lay2(y)
        u, _ = self.lay3(z)
        v = self.lay4(u)
        w, _ = self.lay5(v)
        o = self.lay6(w)
        p, _ = self.lay7(o)
        #q = self.lay8(p)
        return p

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=2, batch_first=True)
        self.lay2 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay5 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay6 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay7 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 1))
        self.drop = nn.Dropout(0.15)

    #forward propagation
    def forward(self, x):
        y, _ = self.lay1(x)
        z = self.lay2(y)
        v, _ = self.lay3(z)
        u = self.lay4(v)
        w, _ = self.lay5(u)
        r = self.lay6(w)
        s, _ = self.lay7(r)
        t = self.lay8(s)
        return torch.sigmoid(t[:, -1])

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def prepareMinutelyData(df:pd.DataFrame, tradingDays:list):
    if df.empty:
        return None
    else:
        ###Some clean up
        df['bfValue'] = df['lastPx']*df['size']
        df['bfValue'] = df['bfValue'].ffill()
        df['cumValue'] = df.groupby('date')['bfValue'].cumsum()
        df=df[df['SP1']>0]
        df=df[df['BP1']>0]
        df=df[df['SP1']-df['BP1']>0]
        for i in range(1, 6):
            df['SP{}'.format(str(i))] = df['SP{}'.format(str(i))]/100
            df['BP{}'.format(str(i))] = df['BP{}'.format(str(i))]/100
            df['SV{}'.format(str(i))] = df['SV{}'.format(str(i))]*1000
            df['BV{}'.format(str(i))] = df['BV{}'.format(str(i))]*1000
        df['lastPx'] = df['lastPx']/100
        df['size'] = df['size']*1000
        df['volume'] = df['volume']*1000
        df['lastPx'] = df.groupby('date')['lastPx'].ffill()
        df['size'] = df.groupby('date')['size'].transform(lambda x: x.fillna(0))
        #df['value'] = df['lastPx'] * df['size']
        df['value'] = df.groupby('date')['cumValue'].diff()
        df['value'] = df['value'].fillna(df['bfValue'])
        del df['bfValue']
        del df['cumValue']
        del df['value']
        
        ###Next, we create datetime, then bin the data to minutely before sending to signal calculation
        df_DateTime = pd.to_datetime(df.date.astype(str) + ' ' + df.time.astype(str), format="%Y-%m-%d %H%M%S%f")
        df['dt_index'] = df_DateTime
        df = df[~df.dt_index.duplicated(keep='last')]        
        
        binSize = '1min'
        
        ###Now, we bin the data to minutely        
        df_minutely = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right')).last()
        for i in range(1, 6):
            df_minutely.loc[:, 'SP{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['SP{}'.format(str(i))].last()
            df_minutely.loc[:, 'BP{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['BP{}'.format(str(i))].last()
            df_minutely.loc[:, 'SV{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['SV{}'.format(str(i))].last()
            df_minutely.loc[:, 'BV{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['BV{}'.format(str(i))].last()
        
        #do some cleaning
        df_minutely = df_minutely.between_time('09:00:00','13:25:00', inclusive = 'right')
        df_minutely['date'] = df_minutely.index.date
        df_minutely['ttime'] = df_minutely.index.time
        #df_minutely['time'].fillna(df_minutely['ttime'], inplace=True)
        df_minutely.fillna({'time': df_minutely['ttime']}, inplace=True)
        del df_minutely['ttime']
        #only keep trading days
        df_minutely = df_minutely[df_minutely['date'].astype(str).isin(tradingDays)]
            
        return df_minutely

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_verge(x, y):
    x = np.mean(x)
    y = np.mean(y)
    return np.sqrt(x ** 2 + y ** 2)

def run_gan(stock: str):
    ###Prepare common directory
    # Use plain string path so string concatenation below works as written
    stockDataDir = str(DATA_DIR) + os.sep
    params_dir = _get_params_output_dir()



    cols = ["date","time","lastPx","size","volume","SP1","BP1","SV1","BV1","SP2","BP2","SV2","BV2","SP3","BP3","SV3","BV3","SP4","BP4","SV4","BV4","SP5","BP5","SV5","BV5"]

    tradingDays = ["2023-10-02","2023-10-03","2023-10-04","2023-10-05","2023-10-06","2023-10-11","2023-10-12","2023-10-13","2023-10-16","2023-10-17","2023-10-18","2023-10-19","2023-10-20","2023-10-23","2023-10-24","2023-10-25","2023-10-26","2023-10-27","2023-10-30","2023-10-31","2023-11-01","2023-11-02","2023-11-03","2023-11-06","2023-11-07","2023-11-08","2023-11-09","2023-11-10","2023-11-13","2023-11-14","2023-11-15","2023-11-16","2023-11-17","2023-11-20","2023-11-21","2023-11-22","2023-11-23","2023-11-24","2023-11-27","2023-11-28","2023-11-29","2023-11-30","2023-12-01","2023-12-04","2023-12-05","2023-12-06","2023-12-07","2023-12-08","2023-12-11","2023-12-12","2023-12-13","2023-12-14","2023-12-15","2023-12-18","2023-12-19","2023-12-20","2023-12-21","2023-12-22","2023-12-25","2023-12-26","2023-12-27","2023-12-28","2023-12-29",
                   "2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08","2024-01-09","2024-01-10","2024-01-11","2024-01-12","2024-01-15","2024-01-16","2024-01-17","2024-01-18","2024-01-19","2024-01-22","2024-01-23","2024-01-24","2024-01-25","2024-01-26","2024-01-29","2024-01-30","2024-01-31","2024-02-01","2024-02-02","2024-02-15","2024-02-16","2024-02-19","2024-02-20","2024-02-21","2024-02-22","2024-02-23","2024-02-26","2024-02-27","2024-02-29"]
    print("Raw data loading and processing " + stock)
        
    ###load stock tick data (gzip)
    file1Path = stockDataDir + stock + '_md_202310_202310.csv.gz'
    file2Path = stockDataDir + stock + '_md_202311_202311.csv.gz'
    file3Path = stockDataDir + stock + '_md_202312_202312.csv.gz'


    df = pd.DataFrame()
    if os.path.exists(file1Path):
        df = pd.concat([df, pd.read_csv(file1Path, compression='gzip', usecols = cols)])
        print('Data 1 for ' + stock + ' loaded.')
    else:
        print(f"Skipping snapshots data {file1Path} for {stock}.")
    if os.path.exists(file2Path):
        df = pd.concat([df, pd.read_csv(file2Path, compression='gzip', usecols = cols)])
        print('Data 2 for ' + stock + ' loaded.')
    else:
        print(f"Skipping snapshots data {file2Path} for {stock}.")
    if os.path.exists(file3Path):
        df = pd.concat([df, pd.read_csv(file3Path, compression='gzip', usecols = cols)])
        print('Data 3 for ' + stock + ' loaded.')
    else:
        print(f"Skipping snapshots data {file3Path} for {stock}.")

    if df.empty:
        print('No order snapshot data loaded. Skipping ' + stock)
        print("No raw data to process; exit.")

        
    else:    
        minutelyData = prepareMinutelyData(df, tradingDays)
        print("Minutely data generated.")
        
        projdata = []
        columns = ['date', 'time', 'lastPx', 'size', 'volume',
                'SP5', 'SP4', 'SP3', 'SP2', 'SP1',
                'BP1', 'BP2', 'BP3', 'BP4', 'BP5',
                'SV5', 'SV4', 'SV3', 'SV2', 'SV1',
                'BV1', 'BV2', 'BV3', 'BV4', 'BV5']
        
        for x in minutelyData.groupby('date'):
            if x[1].shape[0] == 265:
                projdata.append(x[1].values)
    
        projdata = np.array(projdata)    
        
        #normalization
        X = projdata[:,:,5:].astype(float)
        
        X[:,:,-10:] = np.log(1 + X[:,:,-10:])
        X_mean = X.mean(axis=1)
        X_std = X.std(axis=1)
        
        X = np.transpose((np.transpose(X, (1,0,2)) - X_mean) / (2 * X_std), (1,0,2))
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        #set up
        set_seed(307)
        
        generator = Generator()
        discriminator = Discriminator()
        
        #params
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=G_PARAMS['lr'], betas=(0.99, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=D_PARAMS['lr'], betas=(0.99, 0.999))
        
        #batch size
        batch_size = 50
        dataset = MyDataset(torch.tensor(X, dtype=torch.float32))
        
        #training, validation, testing
        train_size = int(0.8 * len(dataset))
        eval_size = int(0.2 * len(dataset))
        train_dataset, eval_dataset = random_split(dataset[:train_size+eval_size], [train_size, eval_size])
        
        #dataloaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True)
        
        #loss functions
        gen_loss = torch.nn.BCELoss()
        loss_function = torch.nn.MSELoss()    
            
        epochs = 200
        
        #storage of loss data
        train_g_loss = [] #
        train_d_loss = [] #
        eval_g_loss = [] #
        eval_d_loss = [] #
        
        train_verge = [] #early stopping
        eval_verge = [] #early stopping
        
        #training starts here
        for epoch in range(epochs):
            for i, data in enumerate(train_dataloader):
        
                #real vs noise
                real = torch.ones(data.size(0), 1)
                fake = torch.zeros(data.size(0), 1)
        
                #train the generator
                generator.train()
                optimizer_G.zero_grad()
        
                gen = generator(data)
        
                d_data = data[:,1:,:] - data[:,:-1,:]
                d_gen = gen[:,1:,:] - gen[:,:-1,:]
                dd_data = d_data[:,1:,:] - d_data[:,:-1,:]
                dd_gen = d_gen[:,1:,:] - d_gen[:,:-1,:]
        
                g_loss = loss_function(discriminator(gen), real) + \
                        loss_function(torch.mean(torch.abs(data), axis=1), torch.mean(torch.abs(gen), axis=1)) + \
                        loss_function(torch.mean(data, axis=1), torch.mean(gen, axis=1)) + \
                        loss_function(torch.mean(data ** 2, axis=1), torch.mean(gen ** 2, axis=1)) + \
                        loss_function(torch.mean(data ** 3, axis=1), torch.mean(gen ** 3, axis=1)) + \
                        loss_function(torch.mean(torch.abs(d_data), axis=1), torch.mean(torch.abs(d_gen), axis=1)) + \
                        loss_function(torch.mean(d_data, axis=1), torch.mean(d_gen, axis=1)) + \
                        loss_function(torch.mean(d_data ** 2, axis=1), torch.mean(d_gen ** 2, axis=1)) +\
                        loss_function(torch.mean(d_data ** 3, axis=1), torch.mean(d_gen ** 3, axis=1)) +\
                        loss_function(torch.mean(torch.abs(dd_data), axis=1), torch.mean(torch.abs(dd_gen), axis=1)) + \
                        loss_function(torch.mean(dd_data, axis=1), torch.mean(dd_gen, axis=1)) + \
                        loss_function(torch.mean(dd_data ** 2, axis=1), torch.mean(dd_gen ** 2, axis=1)) +\
                        loss_function(torch.mean(dd_data ** 3, axis=1), torch.mean(dd_gen ** 3, axis=1))
        
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.3) #clipping the gradient
                optimizer_G.step()
        
                #train the discriminator
                discriminator.train()
                optimizer_D.zero_grad()
        
                real_loss = loss_function(discriminator(data), real)
                fake_loss = loss_function(discriminator(gen.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
        
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.1) #clipping the gradient
                optimizer_D.step()
        
                train_g_loss.append(g_loss.item())
                train_d_loss.append(d_loss.item())
        
            #     if i % 10 == 0:
            #         print("[Epoch %d/%d][Batch %d/%d][D train loss: %f][G train loss: %f]" % (epoch+1, epochs, i+1, len(train_dataloader),
            #                                                                     d_loss.item(), g_loss.item()))
        
            # #validation data set
            g_loss_total = 0
            d_loss_total = 0
            for i, data in enumerate(eval_dataloader):
        
                #real vs. noise
                real = torch.ones(data.size(0), 1)
                fake = torch.zeros(data.size(0), 1)
        
                #evaluate the generator
                generator.eval()
        
                gen = generator(data)
        
                d_data = data[:,1:,:] - data[:,:-1,:]
                d_gen = gen[:,1:,:] - gen[:,:-1,:]
                dd_data = d_data[:,1:,:] - d_data[:,:-1,:]
                dd_gen = d_gen[:,1:,:] - d_gen[:,:-1,:]
        
                g_loss = loss_function(discriminator(gen), real) + \
                        loss_function(torch.mean(torch.abs(data), axis=1), torch.mean(torch.abs(gen), axis=1)) + \
                        loss_function(torch.mean(data, axis=1), torch.mean(gen, axis=1)) + \
                        loss_function(torch.mean(data ** 2, axis=1), torch.mean(gen ** 2, axis=1)) + \
                        loss_function(torch.mean(data ** 3, axis=1), torch.mean(gen ** 3, axis=1)) + \
                        loss_function(torch.mean(torch.abs(d_data), axis=1), torch.mean(torch.abs(d_gen), axis=1)) + \
                        loss_function(torch.mean(d_data, axis=1), torch.mean(d_gen, axis=1)) + \
                        loss_function(torch.mean(d_data ** 2, axis=1), torch.mean(d_gen ** 2, axis=1)) +\
                        loss_function(torch.mean(d_data ** 3, axis=1), torch.mean(d_gen ** 3, axis=1)) +\
                        loss_function(torch.mean(torch.abs(dd_data), axis=1), torch.mean(torch.abs(dd_gen), axis=1)) + \
                        loss_function(torch.mean(dd_data, axis=1), torch.mean(dd_gen, axis=1)) + \
                        loss_function(torch.mean(dd_data ** 2, axis=1), torch.mean(dd_gen ** 2, axis=1)) +\
                        loss_function(torch.mean(dd_data ** 3, axis=1), torch.mean(dd_gen ** 3, axis=1))
                g_loss_total += g_loss
        
                #evaluate the discriminator
                discriminator.eval()
        
                real_loss = loss_function(discriminator(data), real)
                fake_loss = loss_function(discriminator(gen.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss_total += d_loss
        
                eval_g_loss.append(g_loss.item())
                eval_d_loss.append(d_loss.item())
        
                
            # print("[Epoch %d/%d][Batch %d/%d][D eval loss: %f][G eval loss: %f]" % (epoch+1, epochs, i+1, len(eval_dataloader),
            #                                                                     d_loss_total.item()/len(eval_dataloader),
            #                                                                         g_loss_total.item()/len(eval_dataloader)))
        
            train_verge.append(get_verge(train_g_loss[-len(train_dataloader):], train_d_loss[-len(train_dataloader):]))    
            eval_verge.append(get_verge(eval_g_loss[-len(eval_dataloader):], eval_d_loss[-len(eval_dataloader):]))
        
            if epoch >= 5:
                #early stop
                if train_verge[-3] > train_verge[-2] and train_verge[-2] > train_verge[-1] and eval_verge[-3] < eval_verge[-2] and eval_verge[-2] < eval_verge[-1]:
                    break
                
        # # persist training results on training data
        # pd.DataFrame([train_g_loss, train_d_loss], index=['train_g', 'train_d']).to_csv(stock+'_train_g_d.csv')
        # # persist validation results on validation data
        # pd.DataFrame([eval_g_loss, eval_d_loss], index=['eval_g', 'eval_d']).to_csv(stock+'_eval_g_d.csv')
                
        # #persist the model
        # torch.save(generator, stock+'_generator1.pth')
        # torch.save(discriminator, stock+'_discriminator1.pth')
        pd.DataFrame([train_g_loss, train_d_loss], index=['train_g','train_d']).to_csv(params_dir / f"{stock}_train_g_d.csv")
        pd.DataFrame([eval_g_loss, eval_d_loss], index=['eval_g','eval_d']).to_csv(params_dir / f"{stock}_eval_g_d.csv")
        torch.save(generator, params_dir / f"{stock}_generator1.pth")
        torch.save(discriminator, params_dir / f"{stock}_discriminator1.pth")


        
        print("Done training LOB_GAN for stock " + stock)
        
        
        
        # plot loss curves for this stock
        # Save under: ROOT/plots/training_plots/<PARAMS_x>/
        plots_root = ROOT / "plots" / "training_plots"
        plots_root.mkdir(parents=True, exist_ok=True)

        # use the same name as the OUTPUT_DIR subfolder (e.g. PARAMS_1)
        PLOT_DIR = plots_root / params_dir.name
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        
        steps = range(len(train_g_loss))

        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_g_loss, label="Train Generator loss")
        plt.plot(steps, train_d_loss, label="Train Discriminator loss")
        plt.plot(steps, eval_g_loss,  label="Eval Generator loss")
        plt.plot(steps, eval_d_loss,  label="Eval Discriminator loss")

        plt.xlabel("Training step (batch updates)")
        plt.ylabel("Loss")

        plt.title(
            f"GAN Loss Curves — Stock {stock}\n"
            f"G: lr={G_PARAMS['lr']}, betas={G_PARAMS['betas']} | "
            f"D: lr={D_PARAMS['lr']}, betas={D_PARAMS['betas']}"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"{stock}_loss_plot"
            f"_G_lr{G_PARAMS['lr']}_D_lr{D_PARAMS['lr']}.png"
        )
        filepath = PLOT_DIR / filename

        plt.savefig(filepath, dpi=300)
        plt.close()

        print(f"Saved plot: {filepath}")
        print("Done training LOB_GAN for stock " + stock)

    
    
    
if __name__ == '__main__':
        
    stocks = ['2330','0050', '0056']
    for stock in stocks:
        run_gan(stock)
        print(f'completed stock : {stock}')
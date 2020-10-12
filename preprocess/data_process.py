"""
Author      : Tingyu Wang
Description : For preprocess Chinese Sea Surface Temperature index prediction
Version     : 1.1.0
Changelog   : data_process.py 逐层数据处理提取为函数，通过循环处理多层数据
"""

import re
import warnings
import time
import numpy as np
import xarray as xr
import pathlib as pl
from eofs.xarray import Eof
from scipy import signal
import configuration as conf
from meteo_preprocess import \
mask, rm_annual_cycle, float_to_date, date_to_int, rebuild_time, \
pick_nino34, pick_AMO, pick_PDO, pick_customize_index

# %load_ext autoreload
# %autoreload 1
# %aimport configuration

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def Normalization(X):
    """
    [独立功能模块]
    Feature : 标准化数据
    Args    : 
    """
    return ((X - np.mean(X)) / (np.max(X) - np.min(X)))

def get_layer(ds, top = -1, bottom = -1 ):
    """
    [独立功能模块]
    Feature : 计算多层数据平均值
    Args    : 
    """
    ds = ds.loc[:, top: bottom, ...]  
    layer = ds.mean(axis = 1)
    # ds = ds.sel(lev = slice(top, bottom))
    # layer = ds.mean(dim = ['lev'])
    return layer
    
def nc_encoding():
    """
    [独立功能模块]
    Feature : convert Dataset to netcdf file 
              named 'train data' or 'dev data'
    Args    : 
    """
    encoding = {'lat': {
                        '_FillValue': 1e+20,
                        'dtype': 'float32',
                        'zlib': False
                        },
                'lon': {
                        '_FillValue': 1e+20,
                        'dtype': 'float32',
                        'zlib': False
                        },
                'channel': {
                        'dtype': 'int32',
                        'zlib': False
                        },
                'lead_time': {
                        'dtype': 'int32',
                        'zlib': False
                        },
                'sample_num': {
                            'dtype': 'int32',
                            'zlib': False
                            },
                'X':{
                    '_FillValue': 1e+20,
                    'dtype': 'float32',
                    'zlib':False
                    },
                'y':{
                    '_FillValue': 1e+20,
                    'dtype': 'float32',
                    'zlib':False
                    }
            }
    return encoding

def fill_Nan(ds_filter):
    """
    [独立功能模块]
    Feature : 将缺值NaN填补为0.
    Args    : 
    """
    ds_filter[np.isnan(ds_filter)] = 0
    ds_filter = ds_filter.astype(np.float32)
    ds_filter = np.expand_dims(ds_filter, axis = -1)
    # print(ds_filter.dtype)
    return ds_filter

# ================== Class : data_process ==================
class dataprocess():
    def __init__(self, region, *args):
        # print Error if region is not 4 number list
        self.latS, self.latN = region[0], region[1]
        self.lonW, self.lonE = region[2], region[3]
        
    def ds_to_ano(self, ds, vars_name = 'ts', *args):
        """
        *args : layer top, layer bottom
        vars_name : can process sst, ts, thetao ...
        """
        if vars_name == 'ts':
            ds = ds.sel(lat = slice(self.latS, self.latN), 
                        lon = slice(self.lonW, self.lonE))
            ds = mask(ds, 'ocean')
            ds = rebuild_time(ds)
            ds = rm_annual_cycle(ds)
        
        elif vars_name == 'thetao':
            ds = ds.sel(lat = slice(self.latS, self.latN), 
                        lon = slice(self.lonW, self.lonE))
            ds = rebuild_time(ds)
            ds = get_layer(ds, top = 0, bottom = 300)
            ds = rm_annual_cycle(ds)
        
        elif vars_name == 'sst':
            ds = ds.sel(lat = slice(self.latS, self.latN), 
                        lon = slice(self.lonW, self.lonE))
            ds = rm_annual_cycle(ds)
        
        elif vars_name == 'pottmp':
            ds = ds.sel(lat = slice(self.latS, self.latN), 
                        lon = slice(self.lonW, self.lonE))
            ds = get_layer(ds, top = 0, bottom = 300)
            ds = rm_annual_cycle(ds)
        
        else:
            print('Developing...')
            raise
        return ds
        
    def index_calcu(self, SSTA, y_name, region = []):
        # Nino34
        if y_name == 'nino34-index':
            pc = pick_nino34(SSTA)

        # PDO index
        elif y_name == 'PDO-index':
            pc = pick_PDO(SSTA)

        # AMO index
        elif y_name == 'AMO-index':
            pc = pick_AMO(SSTA)

        # Customize SST index
        else:
            pc = pick_customize_index(SSTA, region)

        return pc

class dataload():
    def __init__(self, file_path = '', *args):
        self.file_path = file_path
        
    def main_load(self, vars_name, bgning = 0., ending = 0., 
                  experiment = 'historical', ds_path = ''):
        """
        vars type  : SST, TS, etc.(long data)
        experiment : piControl, historical, Observation(reanalysis)
        """
        if experiment in ['piControl', 'historical']:
            ds = xr.open_dataset(ds_path)[vars_name]\
                   .sel(time = slice(bgning, ending))
        
        elif experiment in ['Observation']:
            
            # deal bgning & ending
            bgning_ym, ending_ym = float_to_date(bgning, ending)
            ds = xr.open_dataset(ds_path)[vars_name]\
                   .sel(time = slice(bgning_ym, ending_ym))
        return ds
        
    # def obey_load(self, vars_name, bgning = 0., ending = 0., 
    #               experiment = 'historical', ds_path = ''):
    #     """
    #     vars type: thetao, uo, vo, wo, etc.(short data)
    #     experiment : piControl, historical, Observation(reanalysis)
    #     """
    #     if experiment in ['historical']:
    #         ds = self.main_load(vars_name, bgning = bgning, ending = ending, 
    #                             experiment = experiment, ds_path = ds_path)
    #     elif experiment in ['piControl']:
    #         pass

def single_layer_process(layer, filter_mode, filter_type):
    # use Butterworth filter
    order = 9
    print(f'Filter Mode: {filter_mode}')
    if filter_mode:
        if filter_type == 'lowpass':
            Wn = 1/3
            filter_time = str(1 / Wn)

        elif filter_type == 'bandpass':
            Wn = [1/96, 1/3]
            filter_time = str(1/Wn[1]) +'-'+ str(1/Wn[0])

        # filter parameters
        b, a = signal.butter(order, Wn, filter_type)
        layer_filt = signal.filtfilt(b, a, layer, axis = 0)

    else:
        print(f'Filter Mode: {filter_mode}')
        Wn = 1
        filter_time = "no-filter"
        layer_filt = layer.data
        
    layer_data = fill_Nan(layer_filt)
    print(f'Layer shape: {layer_data.shape}')
    return layer_data
    
def y_process(pc, filter_mode, filter_type):
    # use Butterworth filter
    order = 9
    if filter_mode:
        if filter_type == 'lowpass':
            Wn = 1/3
            filter_time = str(1 / Wn)

        elif filter_type == 'bandpass':
            Wn = [1/96, 1/3]
            filter_time = str(1/Wn[1]) +'-'+ str(1/Wn[0])

        # filter parameters
        b, a = signal.butter(order, Wn, filter_type)
        label_filt = signal.filtfilt(b, a, pc, axis = 0)

    else:
        print(f'Filter Mode: {filter_mode}')
        Wn = 1
        filter_time = "no-filter"
        label_filt = pc.data
        
    # Normalization
    if conf.y_name == 'PDO-index':
        label_filt = Normalization(label_filt)
        
    label = label_filt.astype(np.float32)
    print(f'Layer shape: {label.shape}')
    return label
    
def main():
    warnings.filterwarnings('ignore')
    # ========================= Config =========================
    if conf.data_type in ['train', 'dev']:
        if conf.hostname == 'ZHAN':
            ts_dir = '../../../CMIP6/historical/Amon/ts/'
            thetao_dir = '../../../CMIP6/historical/Omon/thetao/thetao_'
        elif conf.hostname == 'Group-Server':
            ts_dir = '/disk2/CMIP6/preprocess/output/historical/Amon/ts/'
            thetao_dir = '/disk2/CMIP6/preprocess/output/historical/Omon/thetao/thetao_'
        elif conf.hostname == 'one-model':
            ts_dir = '/disk1/tywang/CMIP6/test/ts/'
            thetao_dir = '/disk1/tywang/CMIP6/test/thetao/thetao_'
        
        ts_name = '*.nc'
        
    elif conf.data_type in ['test', 'real']:
        if conf.hostname == 'Group-Server':
            ts_dir = '/disk1/tywang/data/COBE_SST/'
            thetao_dir = '/disk1/tywang/data/GODAS/GODAS_regrid_mon_1980-2018.nc'
        
        ts_name = 'COBE2_regrid_mon_1850-2019.nc'
    
    # get all CMIP6 piControl data path
    filelist = list(pl.Path(ts_dir).glob(ts_name))
    
    # ========================= Start =========================
    X_, y_ = [], []
    model_list = []
    
    print(f'Data Save Mode: {conf.data_save_mode}')
    
    # ========================= Process each model data =========================
    deal_num = 0
    for i, fpath in enumerate(filelist):
        dl = dataload()
        
        # ========================= Data Loader =========================
        if conf.data_type in ['train', 'dev']:
            # read piControl data
            model_name = re.findall(r".+_(.+?)_.{29}", fpath.name)[0]

            if model_name in conf.model_select:
                print(f"{model_name} will jump!\n")
                continue

            # ts
            ts = dl.main_load(vars_name = conf.layer_vars_name[0], 
                              bgning = conf.bgning, ending = conf.ending, 
                              experiment = conf.experiment, ds_path = fpath)

            # thetao 
            # use try except to avoid no file problem
            thetao_path = thetao_dir + model_name + '_Hist_unigrid_195001-201412.nc'
            try:
                thetao = dl.main_load(vars_name = conf.layer_vars_name[1], 
                                      bgning = conf.bgning, ending = conf.ending, 
                                      experiment = conf.experiment, ds_path = thetao_path)

            except OSError:
                print(f'xxxxxxxx Sorry, {model_name} thetao data does not exist. xxxxxxxx\n')
                continue

            print(f"******** Model name :{model_name} is processing... ********")
            model_list.append(model_name)
    
        elif conf.data_type in ['test', 'real']:
            ts = dl.main_load(vars_name = conf.layer_vars_name[0], 
                              bgning = conf.bgning, ending = conf.ending, 
                              experiment = conf.experiment, ds_path = fpath)
            
            thetao = dl.main_load(vars_name = conf.layer_vars_name[1], 
                                  bgning = conf.bgning, ending = conf.ending, 
                                  experiment = conf.experiment, ds_path = thetao_dir)
            
            model_name = ts_name
            print(f"******** Data name :{model_name} is processing... ********")
            
        # ========================= X - Data Pro =========================    
        # tendency
        tend_ts = dl.main_load(vars_name = conf.layer_vars_name[2], 
                               bgning = conf.tend_bgning, ending = conf.tend_ending,
                               experiment = conf.experiment, ds_path = fpath)
        
        data_pro = dataprocess(region = conf.X_region)
        
        SSTA = data_pro.ds_to_ano(ts, vars_name = conf.layer_vars_name[0])
        print(SSTA)
        
        # Detrend SSTA
        if conf.X_name == 'detrend-SSTA':
            SSTA = SSTA - SSTA.mean(dim = ['lat', 'lon'])
            
        layer1 = data_pro.ds_to_ano(thetao, vars_name = conf.layer_vars_name[1])
        tend_ano = data_pro.ds_to_ano(tend_ts, vars_name = conf.layer_vars_name[2])
        layer2 = SSTA - tend_ano.data  # Tendency
        
        multi_layer_X = [SSTA, layer2]
        if len(multi_layer_X) != conf.layer_num:
            raise Error(f'Need input Layer number is {conf.layer_num}, but get {len(multi_layer_X)}')
        
        # ========================= y - Data Pro =========================
        pc = data_pro.index_calcu(SSTA, conf.y_name, conf.custom_region)
        
        # Loop list
        if conf.data_type == 'train':
            loop_list = range(conf.batch_size * conf.batch_num)
        elif conf.data_type == 'dev':
            loop_list = range(conf.mini_yr, 
                              conf.max_yr - conf.X_len - conf.y_len + 1)
        elif conf.data_type == 'test':
            loop_list = range(conf.max_yr - conf.X_len - conf.y_len + 1)
        elif conf.data_type == 'real':
            loop_list = range(SSTA.sizes['time'])
        else:
            raise Error('No this data type.')
        
        # ========================= filter SSTA / ssta_pdo, index =========================
        
        # X data
        for j in range(conf.layer_num):
        
            # use Butterworth filter
            order = 9
            if conf.filter_mode:
                if conf.filter_type == 'lowpass':
                    Wn = 1/3
                    filter_time = str(1 / Wn)
                elif conf.filter_type == 'bandpass':
                    Wn = [1/96, 1/3]
                    filter_time = str(1/Wn[1]) +'-'+ str(1/Wn[0])
            else:
                Wn = 1
                filter_time = 'no-filter'
            
            multi_layer_X[j] = single_layer_process(multi_layer_X[j], conf.filter_mode, conf.filter_type)
            
        # y data
        label = y_process(pc, conf.filter_mode, conf.filter_type)

        # 切片索引[start: end], end的位置是取不到的
        for m in loop_list:
            print(f"loop num = {m}")
            X_store = np.zeros([1, len(SSTA.lat), len(SSTA.lon), conf.layer_num * conf.X_len])
            # print(f"{m} : {m + X_len}")
            layer_samp = []
            for l in range(conf.layer_num):
                temp = np.swapaxes(multi_layer_X[l][m : m + conf.X_len, ...], 0, 3)
                layer_samp.append(temp)

            # concat multi-layer
            t_len = 0
            for k in np.arange(0, conf.X_len * conf.layer_num, conf.layer_num):
                for l in range(conf.layer_num):
                    # print(layer_samp[l].shape)
                    # print(f"k = {k}")
                    # print(f"l = {l}")
                    # print(f"t_len = {t_len}")
                    X_store[..., k + l] = layer_samp[l][..., t_len] # k: X_len长度循环  l: layer循环
                t_len += 1
            X_.append(X_store)
            
            slice_label = label[ m + conf.X_len : m + conf.X_len + conf.y_len]
            # print(f"{m + X_len} : {m + X_len + y_len}")
            y_.append(slice_label)
        
        deal_num += 1
        print(f'No.{deal_num} model: {model_name} processed.\n')
        # if deal_num == 1:
        #   print(f'For saving space, stop processing, total {deal_num} model.')
        #   break
        
    # ========================= Merge data =========================
    X = np.concatenate(X_, axis = 0)
    y = np.stack(y_, axis = 0)
    print(f'All Dataset: \nX shape: {X.shape} \ny shape: {y.shape}')

    # ========================= split data ========================= 
    if conf.X_len == 3:
        season = ['JFM', 'FMA', 'MAM', 'AMJ', 
                  'MJJ', 'JJA', 'JAS', 'ASO', 
                  'SON', 'OND', 'NDJ', 'DJF']
    elif conf.X_len == 1:
        season = ['Jan', 'Feb', 'Mar', 'Apr',
                  'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec']
    
    encoding = nc_encoding()
    time_log = time.strftime("%Y-%m-%d_%H.%M")
    file_time_log = time.strftime("%m-%d")
    
    if conf.split_type == 'ALL':
        
        # create netCDF data
        sample_num = np.arange(X.shape[0], dtype = 'int32')
        channel = np.arange(X.shape[-1], dtype = 'int32')
        lead_time = np.arange(1, y.shape[-1] + 1, dtype = 'int32')
        lat, lon = SSTA.lat, SSTA.lon

        # set a new xarray Dataset contains train_data:'X', train_label:'y'
        dataset = xr.Dataset(data_vars = {
                                 'X':(['sample_num', 'lat', 'lon', 'channel'], X),
                                 'y':(['sample_num', 'lead_time'], y)
                                 },
                             coords = {
                                 'sample_num': sample_num,
                                 'lat': lat,
                                 'lon': lon,
                                 'channel': channel,
                                 'lead_time': lead_time
                                 })

        # set nc data's attributions
        dataset.attrs['data type'] = conf.data_type
        dataset.attrs['create time'] = f'{time_log} created'
        dataset.attrs['filter'] = f"{conf.filter_type}: {order}-order, Wn = {Wn}, signal: {filter_time} yr"
        dataset.attrs['channel'] = f'{conf.layer_num} layers * {conf.X_len} month'
        dataset.attrs['data source'] = f'{conf.X_name} -> X, {conf.y_name} -> y'
        dataset.attrs['model list'] = model_list
        print(dataset)
        
        if conf.data_save_mode:
            pl.Path(f'../../temp_data/CHN_sea_forecast/{file_time_log}')\
              .mkdir(parents = True, exist_ok=True)
            dataset.to_netcdf(f'../../temp_data/CHN_sea_forecast/{file_time_log}/'+ \
                              conf.data_type +'_'+ filter_time +'_'+ str(conf.y_len) +'-'+ \
                              conf.y_name +'_'+ str(conf.X_len) +'x' + str(conf.layer_num) + \
                              'layer' + '-'+ conf.X_name +'.nc', \
                              encoding = encoding)
        
    elif conf.split_type == '12':
        
        for i, s_name in enumerate(season):
            ds_X = X[i: conf.batch_size*conf.batch_num*deal_num: 12, ...]
            ds_y = y[i: conf.batch_size*conf.batch_num*deal_num: 12, ...]
            # ds_X = X[i:-1:12, ...]  # 不要使用-1，这样倒数第一个数取不到
            # ds_y = y[i:-1:12, ...]
            # print(s_name + ':')
            # print(test_arr[i: batch_size*batch_num: 12])
            
            # create netCDF data
            sample_num = np.arange(ds_X.shape[0], dtype = 'int32')
            channel = np.arange(ds_X.shape[-1], dtype = 'int32')
            lead_time = np.arange(1, ds_y.shape[-1] + 1, dtype = 'int32')
            lat, lon = SSTA.lat, SSTA.lon

            # set a new xarray Dataset contains train_data:'X', train_label:'y'
            dataset = xr.Dataset(data_vars = {
                                     'X':(['sample_num', 'lat', 'lon', 'channel'], ds_X),
                                     'y':(['sample_num', 'lead_time'], ds_y)
                                     },
                                 coords = {
                                     'sample_num': sample_num,
                                     'lat': lat,
                                     'lon': lon,
                                     'channel': channel,
                                     'lead_time': lead_time
                                     })

            # set nc data's attributions
            dataset.attrs['data type'] = conf.data_type
            dataset.attrs['season name'] = f'{s_name}'
            dataset.attrs['create time'] = f'{time_log} created'
            dataset.attrs['filter'] = f"{conf.filter_type}: {order}-order, Wn = {Wn}, signal: {filter_time} yr"
            dataset.attrs['channel'] = f'{conf.layer_num} layers * {conf.X_len} month'
            dataset.attrs['data source'] = f'{conf.X_name} -> X, {conf.y_name} -> y'
            dataset.attrs['model list'] = model_list
            print(dataset, end = '\n\n')
            
            if conf.data_save_mode:
                pl.Path(f'../../temp_data/CHN_sea_forecast/{file_time_log}')\
                  .mkdir(parents = True, exist_ok=True)
                dataset.to_netcdf(f'../../temp_data/CHN_sea_forecast/{file_time_log}/'+ \
                                  s_name + '_' + conf.data_type +'_'+ filter_time +'_'+ \
                                  str(conf.y_len) +'-'+ conf.y_name +'_'+ str(conf.X_len) +'x' + \
                                  str(conf.layer_num) + 'layer' + '-'+ conf.X_name +'.nc', \
                                  encoding = encoding)
    return dataset
                
if __name__ == "__main__":
    dataset = main()   

import time
import numpy as np
import xarray as xr
import pathlib as pl
from eofs.xarray import Eof
from scipy import signal

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

def mask(ds, label='land'):
    """
    [独立功能模块]
    mask陆地或者海洋数据为Nan
    """
    landsea = xr.open_dataset("../../temp_data/mask/landsea.nc")
    landsea = landsea['LSMASK']

    # ds和地形数据分辨率不一致，需将地形数据插值
    # fill_value 使0经度的NaN变为数值，防止出现白边
    landsea = landsea.interp(
                            lat=ds.lat.values, 
                            lon=ds.lon.values,
                            kwargs={'fill_value': None}
                             )

    # 利用地形掩盖海陆数据
    if label == 'ocean':                  # when cond is False
        ds = ds.where(landsea.data == 0)  # default locations fill with NaN
    elif label == 'land':
        ds = ds.where(landsea.data == 1)
    elif label == 'lake':
        ds = ds.where(landsea.data == 2)
    elif label == 'small island':
        ds = ds.where(landsea.data == 3)
    elif label == 'ice shelf':
        ds = ds.where(landsea.data == 4)
    return ds

def rm_annual_cycle(ds):
    """
    [独立功能模块]
    去掉数据年循环（月）
    """
    ds = ds.groupby('time.month') - ds.groupby('time.month').mean()
    ds = ds.reset_coords('month', drop = True)
    return ds

def float_to_date(beginning, ending):
    """
    [独立功能模块]
    Feature : 将float格式日期转为'YYYY-MM'格式
    Args    : 
    """
    bgn_yr, end_yr = int(beginning / 100), int(ending / 100)
    bgn_mon, end_mon = int(beginning - (bgn_yr * 100)), int(ending - (end_yr * 100))
    start_ym = str(bgn_yr).zfill(4) + '-' + str(bgn_mon).zfill(2)
    end_ym = str(end_yr).zfill(4) + '-' + str(end_mon).zfill(2)
    return start_ym, end_ym

def date_to_int(beginning, ending):
    """
    [独立功能模块]
    Feature : 将'YYYY-MM'格式日期转换为int格式
    Args    : 
    """
    if (len(beginning) != 7) or (len(ending) != 7):
        raise InputError("input date is not a 'YYYY-MM' format", f'date length is {len(ending)}')
    bgn_yr, end_yr = beginning[0:4], ending[0:4]
    bgn_mon, end_mon = beginning[5:7], ending[5:7]
    bgn_ym = int(bgn_yr)*100 + int(bgn_mon)
    end_ym = int(end_yr)*100 + int(end_mon)
    return bgn_ym, end_ym

def rebuild_time(ds, calculate_type = None):
    """
    [独立功能模块]
    重构Dataset时间维度
    """
    beginning = ds.time[0]
    ending = ds.time[-1]
    start_ym, end_ym = float_to_date(beginning, ending)
    
    ds['time'] = xr.cftime_range(
                    start = start_ym, 
                    end = end_ym,        # Changelog: change period to end
                    freq = 'MS',
                    calendar = 'noleap'
                    )

    # calculate the year mean
    # ds_end = ds.groupby('time.year').mean()
    # ds_end = ds_end.rename({'year': 'time'})
    ds_end = ds
    return ds_end # 一生的耻辱 原本写成了ds，导致白算了

def pick_nino34(SSTA):
    """
    [独立功能模块]
    计算Nino3.4 index
    """
    nino34 = SSTA.sel(lat = slice(-5, 5), 
                      lon = slice(170, 240)).mean(dim =['lat','lon'])
    return nino34

def pick_AMO(SSTA):
    """
    [独立功能模块]
    计算AMO index
    """
    # SSTA_detrend = signal.detrend(SSTA, axis = 0, type = 'linear')
    AMO = SSTA.sel(lat = slice(0, 65), 
                   lon = slice(280, 360)).mean(dim = ['lat', 'lon'])
    AMO_detrend = signal.detrend(AMO, axis = 0, type = 'linear')
    return AMO_detrend

def pick_PDO(SSTA):
    """
    [独立功能模块]
    计算PDO index
    """
    # to get leading PC1 ==> PDO index
    ssta_pdo = SSTA - SSTA.mean(dim = ['lat', 'lon'])
    ssta_pdo_ = ssta_pdo.sel(lat = slice(20, 70), 
                             lon = slice(110, 260))
    coslat = np.cos(np.deg2rad(ssta_pdo_.coords['lat'].values))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(ssta_pdo_, weights=wgts)

    # no need to calculate EOF pattern
    # eof1 = solver.eofsAsCorrelation(neofs=1)
    pc1 = solver.pcs(npcs=1, pcscaling=1)

    # leading PC
    PDO = pc1[:, 0]
    return PDO

def pick_customize_index(SSTA, region):
    """
    [独立功能模块]
    Feature : 计算任意范围内的温度格点均值指数
    Args    : 
    """
    latS, latN = region[0], region[1]
    lonW, lonE = region[2], region[3]
    print('Customize calculating region:')
    print(f'latS: {latS}, latN: {latN}')
    print(f'lonW: {lonW}, lonE: {lonE}')
    index = SSTA.sel(lat = slice(latS, latN), 
                     lon = slice(lonW, lonE)).mean(dim = ['lat', 'lon'])
    return index


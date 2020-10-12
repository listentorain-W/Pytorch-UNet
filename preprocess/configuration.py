# ==============================================================
#                      Train Set Configuration
# ==============================================================

# experiment        = 'historical' # 'piControl' / 'historical' / 'Observation'

# # 原本batch_size = 62, bgning = 195001.
# # 为了使用tendency故减少一年
# batch_size               = 61
# batch_num                = 12
# bgning, ending           = 195101., 201412.
# tend_bgning, tend_ending = 195011., 201410.

# # Time
# # y_name = 'PDO-index' / 'nino34-index' / 'AMO-index' / 'South China Sea'
# # X_name = 'detrend-SSTA' / 'SSTA'
# X_name, y_name    = 'SSTA', 'South_China_Sea' 
# X_len, y_len      = 1, 12      # time length in X, y Unit: / yr 当y设为0时，代表输出y为空数组，即用于real数据预测
# layer_num         = 3
# layer_vars_name   = ['ts', 'thetao', 'ts']
# mini_yr           = batch_size * batch_num + (X_len + y_len) - 1
# max_yr            = (int(ending / 100) - int(bgning / 100) + 1) * 12

# # region
# #                   latS, latN, lonW, lonE
# X_region          = [-40,   60,   70,  170]
# custom_region     = [  2,   22,  105,  120]
# model_select      = ['E3SM-1-1-ECA', 'E3SM-1-1', 'E3SM-1-0']

# hostname          = 'Group-Server'     # 'ZHAN' / 'Group-Server' / 'one-model'(just read one dataset)
# data_type         = 'train'    # 'train' / 'dev'
# split_type        = '12'       # '12' / 'ALL'
# filter_type       = 'lowpass'  # 'lowpass' / 'bandpass' / 'no_filter'

# # mode
# filter_mode       = True
# data_save_mode    = True


# ==============================================================
#                      Test Set Configuration
# ==============================================================

# experiment        = 'Observation' # 'piControl' / 'historical' / 'Observation'

# # 原本batch_size = 62, bgning = 195001.
# # 为了使用tendency故减少一年
# bgning, ending           = 198101., 201812.
# tend_bgning, tend_ending = 198011., 201810.

# # Time
# # y_name = 'PDO-index' / 'nino34-index' / 'AMO-index' / 'South China Sea'
# # X_name = 'detrend-SSTA' / 'SSTA'
# X_name, y_name    = 'SSTA', 'South_China_Sea' 
# X_len, y_len      = 1, 12      # time length in X, y Unit: / yr  当y设为0时，代表输出y为空数组，即用于real数据预测
# layer_num         = 2
# layer_vars_name   = ['sst', 'pottmp', 'sst']
# max_yr            = (int(ending / 100) - int(bgning / 100) + 1) * 12

# # region
# #                   latS, latN, lonW, lonE
# X_region          = [-40,   60,   70,  170]
# custom_region     = [  2,   22,  105,  120]

# hostname          = 'Group-Server'     # 'ZHAN' / 'Group-Server' / 'one-model'(just read one dataset)
# data_type         = 'test'    # 'test' / 'real'
# split_type        = 'ALL'       # '12' / 'ALL' [test set not support '12' option]
# filter_type       = 'lowpass'  # 'lowpass' / 'bandpass' / 'no_filter'

# # mode
# filter_mode       = True
# data_save_mode    = False

# ==============================================================
#                      Real Set Configuration
# ==============================================================

experiment        = 'Observation' # 'piControl' / 'historical' / 'Observation'

# 原本batch_size = 62, bgning = 195001.
# 为了使用tendency故减少一年
bgning, ending           = 201711., 201803. 
tend_bgning, tend_ending = 201709., 201801.


# Time
# y_name = 'PDO-index' / 'nino34-index' / 'AMO-index' / 'South China Sea'
# X_name = 'detrend-SSTA' / 'SSTA'
X_name, y_name    = 'SSTA', 'East_China_Sea' 
X_len, y_len      = 1, 0      # time length in X, y Unit: / yr 当y设为0时，代表输出y为空数组，即用于real数据预测
layer_num         = 2
layer_vars_name   = ['sst', 'pottmp', 'sst']
max_yr            = (int(ending / 100) - int(bgning / 100) + 1) * 12

# region
#                   latS, latN, lonW, lonE
X_region          = [-40,   60,   70,  170]
custom_region     = [ 23,   34,  117,  131]

hostname          = 'Group-Server'     # 'ZHAN' / 'Group-Server' / 'one-model'(just read one dataset)
data_type         = 'real'    # 'test' / 'real'
split_type        = 'ALL'       # '12' / 'ALL' [test set not support '12' option]
filter_type       = 'lowpass'  # 'lowpass' / 'bandpass' / 'no_filter'

# mode
filter_mode       = False
data_save_mode    = False
if max_yr < 30:
    filter_mode = False # 针对ButterWorth滤波，低于30个样本强制不使用滤波器
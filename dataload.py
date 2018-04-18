import numpy as np
import pandas as pd
import datetime
from torch.utils.data import Dataset


def read_dust(path):
    dust = pd.read_csv(path, engine='python', index_col=0, encoding='utf-8')
    dust.index = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dust.index]
    dust.sort_index(inplace=True)
    return dust


class DustTimeDataset(Dataset):
    
    def __init__(self, dirpath, time_step, y='PM25'):

        CO = self._readmap(dirpath + '/CO.csv', 3.0)
        NO2 = self._readmap(dirpath + '/NO2.csv', 0.12)
        O3 = self._readmap(dirpath + '/O3.csv', 0.16)
        SO2 = self._readmap(dirpath + '/SO2.csv', 0.030)
        PM10 = self._readmap(dirpath + '/PM10.csv', 1000.0, 1.0)
        PM25 = self._readmap(dirpath + '/PM25.csv', 170.0, 1.0)
        기온 = self._readmap(dirpath + '/기온.csv', 37.0, -25.0)
        풍속 = self._readmap(dirpath + '/풍속.csv', 10.0)
        풍향 = self._readmap(dirpath + '/풍향.csv', 360.0)
        강수량 = self._readmap(dirpath + '/강수량.csv', 60.0)
        
        PM_y = read_dust(dirpath + '/' +  y + '.csv')
        PM_y = PM_y.values
        
        data = list(zip(CO, NO2, O3, SO2, PM10, PM25, 기온, 풍향, 풍속, 강수량))
        X = []
        y = []
        for i in range(len(CO) - time_step - 3):
            X.append(data[i: (i+time_step)])
            y.append(PM_y[(i+time_step): (i+time_step+3)])
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        
    def _readmap(self, fpath, max_, min_=0):
        dust = read_dust(fpath)
        dust = (dust - min_) / (max_ - min_)
        coordinates = self._get_coordinates()
        size = dust.shape[0]
        map_ = np.zeros((size, 6, 6))
        
        for i in range(size):
            hour_data = dust.iloc[i]
            for guname, (x, y) in coordinates.items():
                map_[i, x, y] = hour_data[guname]
        
        return map_
    
    def _get_coordinates(self):
        return {'은평구':(0, 1), '강북구': (0, 3), '도봉구': (0, 4),
                '서대문구': (1, 1), '종로구': (1, 2), '성북구': (1, 3), '노원구': (1, 4),
                '강서구': (2, 0), '마포구': (2, 1), '중구': (2, 2), '동대문구': (2, 3), '중랑구': (2, 4),
                '양천구': (3,0), '영등포구': (3, 1), '용산구': (3, 2), '성동구': (3, 3), '광진구': (3, 4), '강동구': (3, 5),
                '구로구': (4, 0), '동작구': (4, 1), '서초구': (4, 2), '강남구': (4, 3), '송파구': (4, 4),
                '금천구': (5, 0), '관악구': (5, 1)}
    
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
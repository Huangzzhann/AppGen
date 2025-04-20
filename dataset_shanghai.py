import pickle
import json
import os
import tqdm
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import DataLoader, Dataset

# app type number
app_type_num = np.arange(20)
# attributes
attributes = ['time', 'app', 'traffic']


def extract_hour(x):
    # extract hour from time: 00:00
    h = datetime.datetime.strptime(str(x), '%Y%m%d%H%M')
    hour = h.hour
    min = h.minute
    t = 2 * hour
    if min > 30:
        t = t + 1
    return t

def extract_minute(x):
    h = datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S')
    hour = h.hour
    min = h.minute
    t = 6 * hour + min // 6

    return t


def trans_to_timestamp(x):
    h = datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S')
    return h.timestamp()


def parse_data(x):
    max_length = 20
    if len(x) == 0:
        return None
    else:
        app_id = x['app'].values.tolist()
        loc_id = x['location'].values.tolist()

        if len(x) < max_length:
            app_id = app_id + [0] * (max_length - len(x))
            loc_id = loc_id + [10083] * (max_length - len(x))

        return app_id[:max_length], loc_id[:max_length]


def parse_id(id_, date):
    data = pd.read_csv('data/shanghai/trace_' + date + '.txt', sep='\t', header=0)

    data = data[data['user'] == id_]
    data['time_bin'] = data['time'].apply(lambda x: extract_minute(str(x)))

    if len(data) == 0:
        return None
    else:
        observed_locs = []
        observed_time = []
        apps = []

        for h in range(144):
            output = parse_data(data[data['time_bin'] == h])
            if output is not None:
                a, l = output
                observed_locs.append(l)
                observed_time.append(h)
                apps.append(a)

        return observed_locs, observed_time, apps


def get_idlist():
    with open('data/shanghai/user2id.json', 'r') as f:
        patient_id = json.load(f)
    return patient_id


class Shanghai_Dataset(Dataset):
    def __init__(self, eval_length=48, mode=None, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_locs = []
        self.observed_time = []
        self.apps = []

        Dates = ['0420', '0421', '0422', '0423', '0424', '0425', '0426']
        # Dates = ['0420']

        path = ("data/shanghai/shanghai" + "_10min" + ".pk")

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            for date in Dates:
                idlist = get_idlist()
                for id_ in tqdm.tqdm(idlist.keys()):
                    try:
                        output = parse_id(int(id_), date)
                        if output is not None:
                            observed_locs, observed_time, apps = output

                            self.observed_locs += observed_locs
                            self.observed_time += observed_time
                            self.apps += apps

                    except Exception as e:
                        print(id_, e)
                        continue

            self.observed_locs = np.array(self.observed_locs)
            self.observed_time = np.array(self.observed_time)
            self.apps = np.array(self.apps)

            self.observed_locs = self.observed_locs.astype("int64")
            self.observed_time = self.observed_time.astype("int64")
            self.apps = self.apps.astype("int64")

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_locs, self.observed_time, self.apps],
                    f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_locs, self.observed_time, self.apps = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_locs))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]

        s = {
            "timepoints": self.observed_time[index],
            "locations": self.observed_locs[index],
            "app_ID": self.apps[index]
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):
    # only to obtain total length of dataset
    dataset = Shanghai_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = Shanghai_Dataset(
        mode='train', use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = Shanghai_Dataset(
        mode='validation', use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = Shanghai_Dataset(
        mode='test', use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path

        if root_path == "./dataset_k" or "./dataset_t":
            self.feature_names = ['0']

        if root_path == "./dataset_k12" or "./dataset_t12":
            self.feature_names = ['0', '1']

        if flag == 'TRAIN': # TODO 正式训练，要打开
            self.root_path = f"{self.root_path}/train"
        elif flag == 'TEST':
            self.root_path = f"{self.root_path}/test" # 注.tyx 忽略传参的影响
        elif flag == 'VAL':
            self.root_path = f"{self.root_path}/val"

        self.flag = flag
        # self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_df, self.labels_df = self.load_all(self.root_path) # 注.tyx
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1) # 注.
        # self.all_IDs = torch.arange(self.all_df.shape[0] // 1000) # 注.tyx.每个样本有1000个数据点
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        # self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # # pre_process 注.tyx 注释掉
        # normalizer = Normalizer()
        # self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_from_numpy(self, load_dir="./"):
        """从NumPy文件加载数据"""
        lc_data = torch.from_numpy(np.load(f"{load_dir}/lc_data.npy"))
        label_data = torch.from_numpy(np.load(f"{load_dir}/label_data.npy"))
        # metadata = np.load(f"{load_dir}/metadata.npy", allow_pickle=True).item()

        # todo
        # lc_data = lc_data[0:60]
        # label_data = label_data[0:60]

        return lc_data, label_data
    def load_kepler_data(self, path):
        test_lc_data, test_label_data = self.load_from_numpy(path)
        X = test_lc_data;
        y = test_label_data

        # 展平数据
        if self.feature_names == ['0']:
            X_flattened = X.reshape(-1, 1)  # todo
        if self.feature_names == ['0', '1']:
            X_flattened = X.reshape(-1, 2)

        labels_df = y.reshape(-1, 1)  # 形状：[112052, 1]

        # 转换为DataFrame
        if self.feature_names == ['0']:
            all_df = pd.DataFrame(X_flattened.numpy(), columns=['dim_0'])

        if self.feature_names == ['0', '1']:
            all_df = pd.DataFrame(X_flattened.numpy(), columns=['dim_0', 'dim_1'])


        # 重新设置索引：每1000行使用同一个样本ID作为索引
        samples_per_timepoint = 512
        new_indices = np.repeat(np.arange(len(labels_df)), samples_per_timepoint)
        all_df.index = new_indices  # 设置新索引

        # 转换标签为DataFrame
        labels_df = pd.DataFrame(labels_df.numpy(), columns=['label'])

        return all_df, labels_df

    def load_all(self, path):
        all_df, labels_df = self.load_kepler_data(path)
        return all_df, labels_df
    def instance_norm(self, case):
        # if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
        #     mean = case.mean(0, keepdim=True)
        #     case = case - mean
        #     stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     case /= stdev
        #     return case
        # else:
        return case

    def __getitem__(self, ind):
        simple_id = self.all_IDs[ind]
        batch_x = self.feature_df.loc[simple_id].values
        labels = self.labels_df.loc[simple_id].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)

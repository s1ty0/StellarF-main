import os
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from sklearn.decomposition import PCA
import math
import torch.nn as nn

def read_parquet_file_kepler(file_path):
    parquet_file = pq.ParquetFile(file_path)
    data = parquet_file.read().to_pandas()

    light_curve = data.flux_norm
    label = data.label
    metadata = data.metadata

    # # todo run
    # light_curve = light_curve[0:10]
    # label = label[0:10]

    return light_curve, label, metadata

# Function Definition - linear interpolate
def linear_interpolate_numpy(t, y):
    """
    Fills missing values using linear interpolation.

    :param t: Time array or index array
    :param y: Light curve data, can be a numpy array, pandas Series, or torch.Tensor
    :return: Interpolated light curve, maintaining the same data type as input
    """
    # 1. Determine the input data type is numpy
    input_type = None
    if isinstance(y, pd.Series):
        input_type = 'series'
        y_values = y.values
        index = y.index
    elif isinstance(y, torch.Tensor):
        input_type = 'tensor'
        y_values = y.numpy()
    else:  # default
        input_type = 'array'
        y_values = y

    # 2. Check if all values are NaN
    if np.all(np.isnan(y_values)):
        raise ValueError("所有数据点均为 NaN，无法进行插值")

    # 3. Perform linear interpolation
    mask = ~np.isnan(y_values)
    t_valid = t[mask]
    y_valid = y_values[mask]

    # Handle the special case where t_valid has only one point (interpolation is not possible)
    if len(t_valid) < 2:
        # Fill with the only valid value
        y_filled = np.full_like(y_values, y_valid[0])
    else:
        y_filled = np.interp(t, t_valid, y_valid)

    # 4. Return results according to the original data type
    if input_type == 'series':
        return pd.Series(y_filled, index=index)
    elif input_type == 'tensor':
        return torch.from_numpy(y_filled)
    else:
        return y_filled

# Function Definition - safe unfold
def safe_unfold(res, dimension, size, step):
    """
    Safely performs the tensor unfold operation, returns None on failure.

    Args:
        res (torch.Tensor): Input tensor.
        dimension (int): Dimension to unfold.
        size (int): Size of each slice.
        step (int): Step between consecutive slices.

    Returns:
        torch.Tensor or None: Unfolded tensor if successful, otherwise None.
    """
    try:
        return res.unfold(dimension=dimension, size=size, step=step)
    except Exception as e:
        print(f"Unfold fail: {e}")
        return None

# Function Definition - patch build
def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    res = xb[:, :tgt_len, :]  # xb: [bs x tgt_len x nvars]

    lc_patch = safe_unfold(res, dimension=1, size=patch_len, step=stride)

    if lc_patch is None:
        print(f"patch_len={patch_len}, stride={stride} contribute to unfold operation failed，pass")
        return None, None

    return lc_patch, num_patch

batch_res = []
# Function Definition - Input & Output Data Matching Function
def match_data(lc, label, patch_len, pred_len, stride):
    lc_patch, num_patch = create_patch(lc, patch_len, stride)

    label_start = pred_len
    label_end = lc.shape[1]

    label_valid = label[:, label_start:label_end, :]
    label_patch, _ = create_patch(label_valid, pred_len, stride) # 步幅一致

    global batch_res # get batch_ids
    if label_patch.shape[1] <= num_patch:  # choose minner
        lc_patch = lc_patch[:, :label_patch.shape[1], :, :]
        batch_res.append(label_patch.shape[1])
    else:
        label_patch = label_patch[:, :num_patch, :, :]
        batch_res.append(num_patch)

    label_patch = torch.any(label_patch == 1, dim=3, keepdim=True).float()
    return lc_patch, label_patch


# Function Definition - data save
def save_as_numpy(lc_tensor, label_tensor, save_dir="./"):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(lc_tensor, torch.Tensor):
        np.save(f"{save_dir}/lc_data.npy", lc_tensor.cpu().numpy())
        np.save(f"{save_dir}/label_data.npy", label_tensor.cpu().numpy())
    else:
        np.save(f"{save_dir}/lc_data.npy", lc_tensor)
        np.save(f"{save_dir}/label_data.npy", label_tensor)
    print(f"Data has saved in {save_dir}")

def normalize_flux(lst):
    """将列表中的值归一化到-1到1的范围"""
    # 计算列表的最小值和最大值
    min_val = min(lst)
    max_val = max(lst)

    # 归一化每个元素
    normalized = []
    for value in lst:
        # 使用公式: 2 * ((x - min) / (max - min)) - 1
        normalized_value = 1 + 0.05 * ((value - min_val) / (max_val - min_val))
        normalized.append(normalized_value)

    return normalized

# Function Definition - process data
def get_std_data(lc_series, label_series, patch_len=512, batch_size=8, pred_len=480, stride=48, save_dir="./", is_tess=False):
    # Three hyperparameters. The default values are the parameters in the original FLARE paper
    patch_len = patch_len
    stride = stride
    pred_len = pred_len

    total_samples = len(lc_series)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_lc_batches = []
    all_label_batches = []

    # Process data in batches
    for batch_idx in range(0, total_samples, batch_size):
        # Get the data of the current batch
        batch_lc = lc_series[batch_idx:batch_idx + batch_size]
        batch_label = label_series[batch_idx:batch_idx + batch_size]

        batch_lc_patches = []
        batch_label_patches = []

        # Process each sample within the batch
        for lc, label in zip(batch_lc, batch_label):
            index = np.arange(len(lc))
            if is_tess:
                lc_nor = normalize_flux(lc)
                lc = np.array(lc_nor)
            else:
                lc = linear_interpolate_numpy(index, lc)

            # Convert to tensor and adjust the shape
            lc_torch = torch.from_numpy(lc)
            lc_input = lc_torch.view(1, -1, 1).to(device)
            label_torch = torch.from_numpy(label)
            label_input = label_torch.view(1, -1, 1).to(device)

            # 生成patch
            lc_patch, label_patch = match_data(lc_input, label_input, patch_len, pred_len, stride)
            batch_lc_patches.append(lc_patch.squeeze(0))  # 去除第0维
            batch_label_patches.append(label_patch.squeeze(0))

        # Concatenate all patches of the current batch
        batch_lc_tensor = torch.cat(batch_lc_patches, dim=0)
        batch_label_tensor = torch.cat(batch_label_patches, dim=0)

        # Store the results of the current batch
        all_lc_batches.append(batch_lc_tensor)
        all_label_batches.append(batch_label_tensor)

        # Clear the gpu memory
        del batch_lc_patches, batch_label_patches, batch_lc_tensor, batch_label_tensor
        torch.cuda.empty_cache()

    # Finally, splice the results of all batches
    final_lc_tensor = torch.cat(all_lc_batches, dim=0)
    final_label_tensor = torch.cat(all_label_batches, dim=0)

    global batch_res
    save_as_numpy(final_lc_tensor, final_label_tensor, save_dir=save_dir)
    return final_lc_tensor, final_label_tensor, batch_res


def build_patch_data_tess(data_dir, save_dir=""):
    data = torch.load(data_dir, weights_only=False)

    light_curve = data['flux']
    label = data['mask']

    # todo run
    # light_curve = light_curve[0:10]
    # label = label[0:10]

    global batch_res
    batch_res = []
    light_curve_std, label_std, batch_res = get_std_data(light_curve, label, save_dir = f"./{save_dir}", is_tess=True)
    return light_curve_std, label_std, batch_res

def build_patch_data_kepler(data_dir, save_dir=""):
    global batch_res
    batch_res = []

    light_curve, label, _ = read_parquet_file_kepler(data_dir)  # get lc and label
    light_curve_train, label_train, batch_res = get_std_data(light_curve, label, patch_len=512, pred_len=480, stride=48,
                                                             save_dir=save_dir)
    return light_curve_train, label_train, batch_res

def get_idea1_txt_tess(data_dir, out_path):
    data = torch.load(data_dir, weights_only=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    light_curve = data['flux']
    label = data['mask']

    # todo run
    # light_curve = light_curve[0:10]
    # label = label[0:10]

    i = 0
    mode = "w"
    for (lc, lb) in zip(light_curve, label):
        flux_median = np.median(lc)
        count_ones = np.sum(lb)
        # line = f"For the star, a total of {count_ones} stellar flares occurred with a median flux of {flux_median}"
        line = f"恒星共计发生了{count_ones}次恒星耀斑，流量中值为{flux_median}"

        with open(out_path, mode, encoding="utf-8") as f:
            f.write(f"{line}\n")
        i = i + 1
        mode = "a"
    print(f"Total deal {i} records have been written.") # to Match.

def get_idea2_txt_kepler(data_dir, out_path):
    light_curve, label, metadata = read_parquet_file_kepler(data_dir)
    print(f"length: {len(label)}")

    # todo run
    # label = label[0:10]

    i = 0
    mode = "w"
    for (lb, md) in zip(label, metadata):
        kic_id = md['kic_id']
        quarter = md['quarter']
        record_list = [i + 1 for i, val in enumerate(lb) if val == 1]

        line = f"恒星id编号{kic_id}，在观测季度{quarter}内的耀斑爆发历史时间点为:{record_list}"
        with open(out_path, mode, encoding="utf-8") as f:
            f.write(f"{line}\n")
        i = i + 1
        mode = "a"

    print(f"Total deal {i} records have been written.")  # to Match.


def get_idea1_txt_kepler(data_dir, out_path):
    light_curve, label, metadata = read_parquet_file_kepler(data_dir)
    print(f"length: {len(label)}")

    # todo run
    # label = label[0:10]
    # metadata = metadata[0:10]

    i = 0
    mode = "w"
    for (lb, md) in zip(label, metadata):
        kic_id = md['kic_id']
        quarter = md['quarter']
        flux_median = md['flux_median']
        count_ones = np.sum(lb)

        line = f"恒星id编号{kic_id}，在观测季度{quarter},共计发生了{count_ones}次恒星耀斑，流量中值为{flux_median}"
        with open(out_path, mode, encoding="utf-8") as f:
            f.write(f"{line}\n")
        i = i + 1
        mode = "a"

    print(f"Total deal {i} records have been written.")  # to Match.


def get_idea2_txt_tess(data_dir, out_path):
    data = torch.load(data_dir, weights_only=False)
    label = data['mask']

    # todo run
    # label = label[0:10]

    i = 0
    mode = "w"
    for lb in label:
        record_list = [i + 1 for i, val in enumerate(lb) if val == 1]
        line = f"恒星在观测季度内的耀斑爆发历史时间点为:{record_list}"
        with open(out_path, mode, encoding="utf-8") as f:
            f.write(f"{line}\n")
        i = i + 1
        mode = "a"

    print(f"Total deal {i} records have been written.")  # to Match.


# Function Definition - batch encode
def batch_encode(texts, model, tokenizer, device, batch_size=32, max_length=128, pooling='cls'):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="batch encode"):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        if pooling == 'cls':
            embeddings = last_hidden[:, 0, :]  # [CLS]标记的表示
        elif pooling == 'mean':
            attention_mask = inputs['attention_mask']
            embeddings = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def get_emb_from_txt(out_path_txt, out_path):
    print("======================================")
    print("Using Bert Encoder to encode our texts.")
    with open(out_path_txt, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    texts = [text.strip() for text in texts] # # 简单预处理（去除换行符）
    # print(texts) # debug
    print(f"Load {len(texts)} records.")
    # model_name = 'bert-base-uncased' # for English(base) # Add some exp.
    model_name = 'bert-base-chinese'                # for Chinese(base)
    # model_name = 'hfl/chinese-bert-wwm'           # BERT pre-trained on Chinese Wikipedia
    # model_name = 'hfl/chinese-roberta-wwm-ext'    # Enhanced Chinese RoBERTa

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # 设置为评估模式

    embeddings = batch_encode(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=32,
        max_length=64,
        pooling='cls'
    )
    print(f"768 shape is: {embeddings.shape}")
    print("Using PCA to turn 768 into 512.")

    pca = PCA(n_components=512) # todo run
    vectors_512 = pca.fit_transform(embeddings)
    print(f"Before: {embeddings.shape}")
    print(f"After PCA: {vectors_512.shape}")
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Proportion of explained variance retained: {explained_variance:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, vectors_512)

def read_parquet_file_kepler(file_path): #
    parquet_file = pq.ParquetFile(file_path)
    data = parquet_file.read().to_pandas()

    light_curve = data.flux_norm
    label = data.label
    metadata = data.metadata

    # # todo run
    # light_curve = light_curve[0:10]
    # label = label[0:10]
    # metadata = metadata[0:10]

    return light_curve, label, metadata


import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


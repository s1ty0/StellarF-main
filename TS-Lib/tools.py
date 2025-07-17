import os
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from sklearn.decomposition import PCA

def read_parquet_file_kepler(file_path):
    parquet_file = pq.ParquetFile(file_path)
    data = parquet_file.read().to_pandas()

    light_curve = data.flux_norm
    label = data.label
    metadata = data.metadata

    # # todo run
    # light_curve = light_curve[0:512]
    # label = label[0:512]

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


# Function Definition - process data
def get_std_data(lc_series, label_series, patch_len=512, batch_size=8, pred_len=480, stride=48, save_dir="./"):
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
    # light_curve = light_curve[0:512]
    # label = label[0:512]

    global batch_res
    batch_res = []
    light_curve_std, label_std, batch_res = get_std_data(light_curve, label, save_dir = f"./{save_dir}")
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
    # light_curve = light_curve[0:512]
    # label = label[0:512]

    i = 0
    mode = "w"
    for (lc, lb) in zip(light_curve, label):
        flux_median = np.median(lc)
        count_ones = np.sum(lb)
        line = f"For the star, a total of {count_ones} stellar flares occurred with a median flux of {flux_median}"
        with open(out_path, mode) as f:
            f.write(f"{line}\n")
        i = i + 1
        mode = "a"
    print(f"Total deal {i} records have been written.") # to Match.

def get_idea2_txt_kepler(data_dir, out_path):
    light_curve, label, metadata = read_parquet_file_kepler(data_dir)
    print(f"length: {len(label)}")

    # todo run
    # label = label[0:512]
    # metadata = metadata[0:512]

    i = 0
    mode = "w"
    for (lb, md) in zip(label, metadata):
        kic_id = md['kic_id']
        quarter = md['quarter']
        record_list = [i + 1 for i, val in enumerate(lb) if val == 1]

        line = f"For the star with ID {kic_id}, during observation quarter {quarter}, the historical time points of flare eruptions during observation quater {quarter} are: {record_list}."
        with open(out_path, mode) as f:
            f.write(f"{line}\n")
        i = i + 1
        mode = "a"

    print(f"Total deal {i} records have been written.")  # to Match.


def get_idea1_txt_kepler(data_dir, out_path):
    light_curve, label, metadata = read_parquet_file_kepler(data_dir)
    print(f"length: {len(label)}")

    # todo run
    # label = label[0:512]
    # metadata = metadata[0:512]

    i = 0
    mode = "w"
    for (lb, md) in zip(label, metadata):
        kic_id = md['kic_id']
        quarter = md['quarter']
        flux_median = md['flux_median']
        count_ones = np.sum(lb)

        line = f"For the star with ID {kic_id}, during observation quarter {quarter}, a total of {count_ones} stellar flares occurred with a median flux of {flux_median}."
        with open(out_path, mode) as f:
            f.write(f"{line}\n")
        i = i + 1
        mode = "a"

    print(f"Total deal {i} records have been written.")  # to Match.


def get_idea2_txt_tess(data_dir, out_path):
    data = torch.load(data_dir, weights_only=False)
    label = data['mask']

    # todo run
    # label = label[0:512]

    i = 0
    mode = "w"
    for lb in label:
        record_list = [i + 1 for i, val in enumerate(lb) if val == 1]
        line = f"For the star, the historical time points of flare eruptions during observation quarter are {record_list}"
        with open(out_path, mode) as f:
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
    model_name = 'bert-base-uncased' # for English(base) # Add some exp.
    # model_name = 'bert-base-chinese'                # for Chinese(base)
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

    # todo run
    # light_curve = light_curve[0:512]
    # label = label[0:512]
    # metadata = metadata[0:512]

    return light_curve, label, metadata

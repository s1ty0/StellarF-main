import os

import numpy as np
import torch

from tools import get_idea1_txt_tess, get_emb_from_txt, build_patch_data_tess, get_idea2_txt_tess, save_as_numpy, \
    read_parquet_file_kepler, get_idea1_txt_kepler, get_idea2_txt_kepler

os.environ['HF_TOKEN'] = "your_token_here" # todo

"""
    Kepler data's idea1 & idea2.
"""
# 1. Get the idea1's data.
# GET IDEA1 TXT
print("Getting IDEA1 txt, from train test val")
train_data_dir = r"raw_data/kepler/train-00000-of-00001.parquet"
out_path_train = r"dataset_k12/idea1_kepler_train.txt"
get_idea1_txt_kepler(train_data_dir, out_path_train)

test_data_dir = r"raw_data/kepler/test-00000-of-00001.parquet"
out_path_test = r"dataset_k12/idea1_kepler_test.txt"
get_idea1_txt_kepler(test_data_dir, out_path_test)

val_data_dir = r"raw_data/kepler/validation-00000-of-00001.parquet"
out_path_val = r"dataset_k12/idea1_kepler_val.txt"
get_idea1_txt_kepler(val_data_dir, out_path_val)

# 2.
print("Turn txt into 512emb (train)")
out_path_emb_train  = r"dataset_k12/idea1_emb_train.npy"
get_emb_from_txt(out_path_train, out_path_emb_train)
print("ok")

print("Turn txt into 512emb (test)")
out_path_emb_test  = r"dataset_k12/idea1_emb_test.npy"
get_emb_from_txt(out_path_test, out_path_emb_test)
print("ok")

print("Turn txt into 512emb (val)")
out_path_emb_val  = r"dataset_k12/idea1_emb_val.npy"
get_emb_from_txt(out_path_val, out_path_emb_val)
print("ok")

# GET IDEA2 TXT
print("=============================================")
print("Getting IDEA2 txt, from train test val")
train_data_dir = r"raw_data/kepler/train-00000-of-00001.parquet"
out_path_train = r"dataset_k12/idea2_kepler_train.txt"
get_idea2_txt_kepler(train_data_dir, out_path_train)

test_data_dir = r"raw_data/kepler/test-00000-of-00001.parquet"
out_path_test = r"dataset_k12/idea2_kepler_test.txt"
get_idea2_txt_kepler(test_data_dir, out_path_test)

val_data_dir = r"raw_data/kepler/validation-00000-of-00001.parquet"
out_path_val = r"dataset_k12/idea2_kepler_val.txt"
get_idea2_txt_kepler(val_data_dir, out_path_val)

# 2.
print("Turn txt into 512emb (train)")
out_path_emb_train  = r"dataset_k12/idea2_emb_train.npy"
get_emb_from_txt(out_path_train, out_path_emb_train)

print("Turn txt into 512emb (test)")
out_path_emb_test  = r"dataset_k12/idea2_emb_test.npy"
get_emb_from_txt(out_path_test, out_path_emb_test)

print("Turn txt into 512emb (val)")
out_path_emb_val  = r"dataset_k12/idea2_emb_val.npy"
get_emb_from_txt(out_path_val, out_path_emb_val)


print("All data done, begin to fuse!")
print("=============================================")
print("TRAIN.........")
lc_data_train = r"dataset_k/train/lc_data.npy" # have patched
patch_ids_train = r"dataset_k12/patch_ids_train.npy"
idea1_emb_train = r"dataset_k12/idea1_emb_train.npy" # no patched
idea2_emb_train = r"dataset_k12/idea2_emb_train.npy"

print("======================================")
print("TRAIN IDEA1 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_train))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb1 = torch.from_numpy(np.load(idea1_emb_train))
print("emb's shape: ",emb1.shape)

expanded_emb1 = torch.repeat_interleave(emb1, patch_nums1, dim=0)
expanded_emb1_end = expanded_emb1.unsqueeze(1) + torch.from_numpy(np.load(lc_data_train))
# expanded_emb1_end = torch.cat([expanded_emb1.unsqueeze(1), torch.from_numpy(np.load(lc_data_train))], dim=1)

train_lc_data = r"dataset_k12/train/lc_data.npy"
np.save(train_lc_data, expanded_emb1_end)
print("TRAIN IDEA1 FUSION OK AND SAVED IN dataset_k12/train/.")
del emb1
del expanded_emb1
del expanded_emb1_end

print("======================================")
print("TRAIN IDEA2 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_train))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb2 = torch.from_numpy(np.load(idea2_emb_train))
print("emb's shape: ",emb2.shape)

expanded_emb2 = torch.repeat_interleave(emb2, patch_nums1, dim=0)
expanded_emb2_end = torch.cat([expanded_emb2.unsqueeze(1), torch.from_numpy(np.load(lc_data_train))], dim=1)

train_lc_data = r"dataset_k12/train/lc_data.npy"
np.save(train_lc_data, expanded_emb2_end)
print("TRAIN IDEA2 FUSION OK AND SAVED IN dataset_k12/train/.")
del emb2
del expanded_emb2
del expanded_emb2_end

print("=============================================")
print("TEST.........DUDU")
lc_data_train = r"dataset_k/test/lc_data.npy" # have patched
patch_ids_train = r"dataset_k12/patch_ids_test.npy"
idea1_emb_train = r"dataset_k12/idea1_emb_test.npy" # no patched
idea2_emb_train = r"dataset_k12/idea2_emb_test.npy"

print("======================================")
print("TEST IDEA1 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_train))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb1 = torch.from_numpy(np.load(idea1_emb_train))
print("emb's shape: ",emb1.shape)

expanded_emb1 = torch.repeat_interleave(emb1, patch_nums1, dim=0)
expanded_emb1_end = expanded_emb1.unsqueeze(1) + torch.from_numpy(np.load(lc_data_train))
# expanded_emb1_end = torch.cat([expanded_emb1.unsqueeze(1), torch.from_numpy(np.load(lc_data_train))], dim=1)

train_lc_data = r"dataset_k12/test/lc_data.npy"
np.save(train_lc_data, expanded_emb1_end)
print("TEST IDEA1 FUSION OK AND SAVED IN dataset_k12/test/.")
del emb1
del expanded_emb1
del expanded_emb1_end

print("======================================")
print("TEST IDEA2 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_train))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb2 = torch.from_numpy(np.load(idea2_emb_train))
print("emb's shape: ",emb2.shape)

expanded_emb2 = torch.repeat_interleave(emb2, patch_nums1, dim=0)
expanded_emb2_end = torch.cat([expanded_emb2.unsqueeze(1), torch.from_numpy(np.load(lc_data_train))], dim=1)

train_lc_data = r"dataset_k12/test/lc_data.npy"
np.save(train_lc_data, expanded_emb2_end)
print("TRAIN IDEA2 FUSION OK AND SAVED IN dataset_k12/test/.")
del emb2
del expanded_emb2
del expanded_emb2_end


print("=============================================")
print("VAL.........DUDU")
lc_data_train = r"dataset_k/val/lc_data.npy" # have patched
patch_ids_train = r"dataset_k12/patch_ids_val.npy"
idea1_emb_train = r"dataset_k12/idea1_emb_val.npy" # no patched
idea2_emb_train = r"dataset_k12/idea2_emb_val.npy"

print("======================================")
print("VAL IDEA1 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_train))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb1 = torch.from_numpy(np.load(idea1_emb_train))
print("emb's shape: ",emb1.shape)

expanded_emb1 = torch.repeat_interleave(emb1, patch_nums1, dim=0)
expanded_emb1_end = expanded_emb1.unsqueeze(1) + torch.from_numpy(np.load(lc_data_train))
# expanded_emb1_end = torch.cat([expanded_emb1.unsqueeze(1), torch.from_numpy(np.load(lc_data_train))], dim=1)

train_lc_data = r"dataset_k12/val/lc_data.npy"
np.save(train_lc_data, expanded_emb1_end)
print("VAL IDEA1 FUSION OK AND SAVED IN dataset_k12/val/.")
del emb1
del expanded_emb1
del expanded_emb1_end

print("======================================")
print("VAL IDEA2 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_train))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb2 = torch.from_numpy(np.load(idea2_emb_train))
print("emb's shape: ",emb2.shape)

expanded_emb2 = torch.repeat_interleave(emb2, patch_nums1, dim=0)
expanded_emb2_end = torch.cat([expanded_emb2.unsqueeze(1), torch.from_numpy(np.load(lc_data_train))], dim=1)

train_lc_data = r"dataset_k12/val/lc_data.npy"
np.save(train_lc_data, expanded_emb2_end)
print("TRAIN IDEA2 FUSION OK AND SAVED IN dataset_k12/train/.")
del emb2
del expanded_emb2
del expanded_emb2_end

print("Kepler DONE")
print("And this dataset has been saved in dataset_k12/train test val")

"""
    Tess data's idea1 & idea2.
"""
# 1. Get the idea1's data.
# SOURCE 1
print("======================================")
print("IDEA1....................................")
print("SOURCE1....................................")
print("Load (1) patch data....................................")
test_data_path = r"raw_data/tess/test.pt"
save_dir = "dataset_t12/source1"
test_lc_source1, test_label_source2, batch_res = build_patch_data_tess(test_data_path, save_dir = save_dir)

print("Data source1 X's shape is ", test_lc_source1.shape)
print("Data source1 y's shape is ", test_label_source2.shape)
print("load patch data ok.")
del test_lc_source1, test_label_source2

print("load (2) patch_ids....................................")
np_array = np.array(batch_res)
print("batch_res: ", np_array.shape)
print("sum: ", sum(batch_res))
np.save('dataset_t12/patch_ids_source1.npy', np_array)
print("load patch_ids ok.")
del batch_res
del np_array

print("load (3) embs....................................")
out_path_txt = r"./dataset_t12/idea1_source1.txt"
get_idea1_txt_tess(test_data_path, out_path_txt)

print("Turn txt into 512emb from source 1.")
out_path_emb_sourse1  = r"dataset_t12/idea1_emb_source1.npy"
get_emb_from_txt(out_path_txt, out_path_emb_sourse1)
print("load embs ok.")

print("SOURCE1 data ok.")

# SOURCE 2
print("======================================")
print("SOURCE2.................................... ")
print("Load (1) patch data....................................")
val_data_path = r"raw_data/tess/valid.pt"
save_dir = "dataset_t12/source2"
val_lc_source1, val_label_source2, batch_res = build_patch_data_tess(val_data_path, save_dir = save_dir)

print("Data source1 X's shape is ", val_lc_source1.shape)
print("Data source1 y's shape is ", val_label_source2.shape)
print("load patch data ok.")
del val_lc_source1, val_label_source2

print("load (2) patch_ids....................................")
np_array = np.array(batch_res)
print("batch_res: ", np_array.shape)
print("sum: ", sum(batch_res))
np.save('dataset_t12/patch_ids_source2.npy', np_array)
print("load patch_ids ok.")
del batch_res
del np_array

print("load (3) embs....................................")
out_path_txt = r"./dataset_t12/idea1_source2.txt"
get_idea1_txt_tess(val_data_path,out_path_txt)

print("Turn txt into 512emb from source 2.")
out_path_emb_sourse1  = r"dataset_t12/idea1_emb_source2.npy"
get_emb_from_txt(out_path_txt, out_path_emb_sourse1)
print("load embs ok.")

print("SOURCE2 data ok.")

print("======================================")
print("Now we have collect all about idea1's data. There is 3: ")
print("(1) patch light data")
print("(2) emb_source1 & emb_source2")
print("(3) patch_ids_source1 & patch_ids_source2")
print("Now we begin to fusion there datas.")

lc_data_source1 = r"dataset_t12/source1/lc_data.npy" # have patched
lc_data_source2 = r"dataset_t12/source2/lc_data.npy"

patch_ids_source1 = r"dataset_t12/patch_ids_source1.npy"
patch_ids_source2 = r"dataset_t12/patch_ids_source2.npy"

idea1_emb_source1 = r"dataset_t12/idea1_emb_source1.npy" # no patched
idea1_emb_source2 = r"dataset_t12/idea1_emb_source2.npy"

print("======================================")
print("SOURCE1 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_source1))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb1 = torch.from_numpy(np.load(idea1_emb_source1))
print("emb's shape: ",emb1.shape)

expanded_emb1 = torch.repeat_interleave(emb1, patch_nums1, dim=0)
expanded_emb1_end = expanded_emb1.unsqueeze(1) + torch.from_numpy(np.load(lc_data_source1))

source1_lc_data = r"dataset_t12/source1/lc_data.npy"
np.save(source1_lc_data, expanded_emb1_end)
print("SOURCE1 DATA FUSION OK AND SAVED IN dataset_t12/source1/.")
del emb1, patch_nums1
del expanded_emb1
del expanded_emb1_end # we have save, del it to free memory

print("SOURCE2 DATA FUSION")
patch_nums2 = torch.from_numpy(np.load(patch_ids_source2))
print("patch_nums's shape: ",patch_nums2.shape)
print("total samples: ",sum(patch_nums2))

emb2 = torch.from_numpy(np.load(idea1_emb_source2))
print("emb's shape: ",emb2.shape)

expanded_emb2 = torch.repeat_interleave(emb2, patch_nums2, dim=0)
emb2_tmp = expanded_emb2.unsqueeze(1)
lc_tmp = torch.from_numpy(np.load(lc_data_source2))
print("Before concat: shape1 is", emb2_tmp.shape, "shape2 is", lc_tmp.shape)
expanded_emb2_end =  emb2_tmp + lc_tmp

source2_lc_data = r"dataset_t12/source2/lc_data.npy"
np.save(source2_lc_data, expanded_emb2_end)
print("SOURCE2 DATA FUSION OK AND SAVED IN dataset_t12/source2/")
del emb2, patch_nums2
del expanded_emb2
del emb2_tmp, lc_tmp
del expanded_emb2_end

# # 2. idea2 data
print("======================================")
print("Now we have collect all about idea1's data. Then we begin to collect idea2's data and fusion there into final data. ")
# SOURCE 1
print("======================================")
print("SOURCE1.................................... ")

print("we have load (1) and (2) data....................................")
print("now we load (3) embs....................................")
out_path_txt = r"./dataset_t12/idea2_source1.txt"
get_idea2_txt_tess(test_data_path,out_path_txt)

print("Turn txt into 512emb from source 1.")
out_path_emb_sourse1  = r"dataset_t12/idea2_emb_source1.npy"
get_emb_from_txt(out_path_txt, out_path_emb_sourse1)
print("load embs ok.")
print("SOURCE1 data ok.")


val_data_path = r"raw_data/tess/valid.pt"
# SOURCE 2
print("======================================")
print("SOURCE2.................................... ")

print("we have load (1) and (2) data....................................")
print("load (3) embs....................................")
out_path_txt = r"./dataset_t12/idea2_source2.txt"
get_idea2_txt_tess(val_data_path,out_path_txt)

print("Turn txt into 512emb from source 2.")
out_path_emb_sourse1  = r"dataset_t12/idea2_emb_source2.npy"
get_emb_from_txt(out_path_txt, out_path_emb_sourse1)
print("load embs ok.")
print("SOURCE2 data ok.")

print("======================================")
print("Now we have collect all about idea1's and idea2's data. ")
print("Now we begin to fusion final datas.")

lc_data_source1 = r"dataset_t12/source1/lc_data.npy" # have patched
lc_data_source2 = r"dataset_t12/source2/lc_data.npy"

patch_ids_source1 = r"dataset_t12/patch_ids_source1.npy"
patch_ids_source2 = r"dataset_t12/patch_ids_source2.npy"

idea2_emb_source1 = r"dataset_t12/idea2_emb_source1.npy" # no patched
idea2_emb_source2 = r"dataset_t12/idea2_emb_source2.npy"

print("======================================")
print("SOURCE1 DATA FUSION")
patch_nums1 = torch.from_numpy(np.load(patch_ids_source1))
print("patch_nums's shape: ",patch_nums1.shape)
print("total samples: ",sum(patch_nums1))

emb1 = torch.from_numpy(np.load(idea2_emb_source1))
print("emb's shape: ",emb1.shape)

expanded_emb1 = torch.repeat_interleave(emb1, patch_nums1, dim=0)
expanded_emb1_end = torch.cat([expanded_emb1.unsqueeze(1), torch.from_numpy(np.load(lc_data_source1))], dim=1) # idea2 最终拼接

source1_lc_data = r"dataset_t12/source1/lc_data.npy"
np.save(source1_lc_data, expanded_emb1_end)
print("SOURCE1 DATA FUSION OK AND SAVED IN dataset12/source1/.")
del emb1
del expanded_emb1
del expanded_emb1_end

print("SOURCE2 DATA FUSION")
patch_nums2 = torch.from_numpy(np.load(patch_ids_source2))
print("patch_nums's shape: ",patch_nums2.shape)
print("total samples: ",sum(patch_nums2))

emb2 = torch.from_numpy(np.load(idea2_emb_source2))
print("emb's shape: ",emb2.shape)

expanded_emb2 = torch.repeat_interleave(emb2, patch_nums2, dim=0)
expanded_emb2_end = torch.cat([expanded_emb2.unsqueeze(1), torch.from_numpy(np.load(lc_data_source2))], dim=1)

source2_lc_data = r"dataset_t12/source2/lc_data.npy"
np.save(source2_lc_data, expanded_emb2_end)
print("SOURCE2 DATA FUSION OK AND SAVED IN dataset12/source2/")
del emb2
del expanded_emb2
del expanded_emb2_end

print("FUSION OK!! NOW WE BEGIN TO SPLIT THEM.")
print("Notably, we only need to split these data according to before settings.")

print("======================================")
print("Build test data from source 1 & 2 data")
test_lc = np.load(r"dataset_t12/source1/lc_data.npy")
test_label = np.load(r"dataset_t12/source1/label_data.npy")

# # todo del
# test_lc = test_lc[0:100]
# test_label = test_label[0:100]

val_lc = np.load(r"dataset_t12/source2/lc_data.npy")
val_label = np.load(r"dataset_t12/source2/label_data.npy")

# # todo del
# val_lc = val_lc[0:100]
# val_label = val_label[0:100]


# build test data
take_num = len(test_lc)
val_num = len(val_lc)
sample_num = take_num + val_num

train_num = int(sample_num * 0.6)
test_num = int(sample_num * 0.3)
val_num = sample_num - train_num - test_num
print("train samples: ", train_num)
print("test samples: ", test_num)
print("val samples: ", val_num)

data_lc_test = test_lc[:test_num]  # 取出要叠加的数据
data_label_test = test_label[:test_num]  # 取出要叠加的数据
save_as_numpy(data_lc_test, data_label_test, save_dir="dataset_t12/test")
print("Test data X's shape is ", data_lc_test.shape)
print("Test data y's shape is ", data_label_test.shape)
del data_lc_test, data_label_test

print("Build val data from source 1 & 2 data")
data_lc_val = val_lc[:val_num]
data_label_val = val_label[:val_num]
save_as_numpy(data_lc_val, data_label_val, save_dir="dataset_t12/val")
print("Val data X's shape is ", data_lc_val.shape)
print("Val data y's shape is ", data_label_val.shape)
del data_lc_val, data_label_val

print("Build train data from source 1 & 2 data")
left_lc_test = test_lc[test_num:]
left_label_test = test_label[test_num:]
left_lc_val = val_lc[val_num:]
left_label_val = val_label[val_num:]

train_lc = np.vstack([left_lc_val, left_lc_test])  # Overlay
train_label = np.vstack([left_label_val, left_label_test])
save_as_numpy(train_lc, train_label, save_dir="dataset_t12/train")
print("Train data X's shape is ", train_lc.shape)
print("Train data y's shape is ", train_label.shape)
print("Shape is: (patch_num, dim, d_model)")
del train_lc, train_label
print("Tess dataset which added 1&2 has been saved in dataset_t12!")
del test_label, test_lc, val_label, val_lc
del left_lc_test, left_label_test, left_lc_val, left_label_val
del take_num, sample_num, train_num, test_num, val_num
del idea1_emb_source1, idea1_emb_source2, idea2_emb_source1, idea2_emb_source2
del lc_data_source1, lc_data_source2, patch_ids_source1, patch_ids_source2
del save_dir, out_path_emb_sourse1, out_path_txt
del source1_lc_data, source2_lc_data
del test_data_path, val_data_path

print("======================================")

print("Balance data....................................")
test_lc_path = 'dataset_t12/test/lc_data.npy'
test_label_path = 'dataset_t12/test/label_data.npy'

val_lc_path = 'dataset_t12/val/lc_data.npy'
val_label_path = 'dataset_t12/val/label_data.npy'

train_lc_path = 'dataset_t12/train/lc_data.npy'
train_label_path = 'dataset_t12/train/label_data.npy'

print("Balance val data....................................")
val_lc = np.load(val_lc_path)
val_label = np.load(val_label_path)

count_0 = np.count_nonzero(val_label == 0)
count_1 = np.count_nonzero(val_label == 1)

negative_indices = np.where(val_label == 0)[0]
positive_indices = np.where(val_label == 1)[0]

target_negatives = int(len(positive_indices) * 1.3)

selected_negatives = negative_indices[:target_negatives]

selected_indices = np.concatenate([selected_negatives, positive_indices])

balanced_lc = val_lc[selected_indices]
balanced_label = val_label[selected_indices]

new_count_0 = np.count_nonzero(balanced_label == 0)
new_count_1 = np.count_nonzero(balanced_label == 1)
print(f"Negative sample count: {new_count_0}, positive sample count: {new_count_1}")

np.save(val_lc_path, balanced_lc)
np.save(val_label_path, balanced_label)

del val_lc, val_label
del balanced_lc, balanced_label
del selected_indices, selected_negatives
del negative_indices, positive_indices

print("Balance test data....................................")
val_lc = np.load(test_lc_path)
val_label = np.load(test_label_path)

count_0 = np.count_nonzero(val_label == 0)
count_1 = np.count_nonzero(val_label == 1)

negative_indices = np.where(val_label == 0)[0]
positive_indices = np.where(val_label == 1)[0]

target_negatives = int(len(positive_indices) * 1.3)

selected_negatives = negative_indices[:target_negatives]

selected_indices = np.concatenate([selected_negatives, positive_indices])

balanced_lc = val_lc[selected_indices]
balanced_label = val_label[selected_indices]

new_count_0 = np.count_nonzero(balanced_label == 0)
new_count_1 = np.count_nonzero(balanced_label == 1)
print(f"Negative sample count: {new_count_0}, positive sample count: {new_count_1}")

np.save(test_lc_path, balanced_lc)
np.save(test_label_path, balanced_label)

del val_lc, val_label
del balanced_lc, balanced_label
del selected_indices, selected_negatives
del negative_indices, positive_indices

print("Balance train data....................................")
val_lc = np.load(train_lc_path)
val_label = np.load(train_label_path)

count_0 = np.count_nonzero(val_label == 0)
count_1 = np.count_nonzero(val_label == 1)

negative_indices = np.where(val_label == 0)[0]
positive_indices = np.where(val_label == 1)[0]

target_negatives = int(len(positive_indices) * 1.3)

selected_negatives = negative_indices[:target_negatives]

selected_indices = np.concatenate([selected_negatives, positive_indices])

balanced_lc = val_lc[selected_indices]
balanced_label = val_label[selected_indices]

new_count_0 = np.count_nonzero(balanced_label == 0)
new_count_1 = np.count_nonzero(balanced_label == 1)
print(f"Negative sample count: {new_count_0}, positive sample count: {new_count_1}")

print("train data shape is: ", balanced_lc.shape)
np.save(train_lc_path, balanced_lc)
np.save(train_label_path, balanced_label)

del val_lc, val_label
del balanced_lc, balanced_label
del selected_indices, selected_negatives
del negative_indices, positive_indices

del train_lc_path, train_label_path
del test_lc_path, test_label_path
del val_lc_path, val_label_path
del count_0, count_1
del new_count_0, new_count_1
del target_negatives

print("Work Over, now you have the final tess dataset with idea1 & idea2. ")
print("And this dataset has been saved in dataset_t12/train test val")

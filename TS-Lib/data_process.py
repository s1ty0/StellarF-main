import numpy as np
import torch

from tools import read_parquet_file_kepler, get_std_data, build_patch_data_tess, save_as_numpy, build_patch_data_kepler

# """
#     一、Kepler dataset build.
# """
print("Begin is the Kepler Light Curve.")
print("======================================")
print("Kepler train data build...")

train_data_dir = r"raw_data/kepler/train-00000-of-00001.parquet"
save_dir_train = "dataset_k/train"

light_curve_train, label_train, batch_res = build_patch_data_kepler(train_data_dir, save_dir_train)
print("Kepler train dataset X's shape is ", light_curve_train.shape)
print("Kepler train dataset y's shape is ", label_train.shape)
print("Shape is: (patch_num, dim, d_model)")

np_array = np.array(batch_res)
print("batch_res: ", np_array.shape)
print("sum: ", sum(batch_res))
np.save('dataset_k12/patch_ids_train.npy', np_array)
del light_curve_train, label_train

print("======================================")
print("Kepler test data build...")
test_data_dir = r"raw_data/kepler/test-00000-of-00001.parquet"
save_dir_test = "dataset_k/test"

light_curve_test, label_test, batch_res = build_patch_data_kepler(test_data_dir, save_dir_test)
print("Kepler test dataset X's shape is ", light_curve_test.shape)
print("Kepler test dataset y's shape is ", label_test.shape)
print("Shape is: (patch_num, dim, d_model)")

np_array = np.array(batch_res)
print("batch_res: ", np_array.shape)
print("sum: ", sum(batch_res))
np.save('dataset_k12/patch_ids_test.npy', np_array)
del light_curve_test, label_test

print("======================================")
print("Kepler val data build...")
val_data_dir = r"raw_data/kepler/validation-00000-of-00001.parquet"
save_dir_val = "dataset_k/val"

light_curve_val, label_val, batch_res = build_patch_data_kepler(val_data_dir, save_dir_val)
print("Kepler val dataset X's shape is ", light_curve_val.shape)
print("Kepler val dataset y's shape is ", label_val.shape)
print("Shape is: (patch_num, dim, d_model)")

np_array = np.array(batch_res)
print("batch_res: ", np_array.shape)
print("sum: ", sum(batch_res))
np.save('dataset_k12/patch_ids_val.npy', np_array)
del light_curve_val, label_val

print("======================================")
print("Kepler dataset which saved in dataset_k built over!")

"""
    二、Tess dataset build. # attn. wo choose 2 source data to build trian, val and test.
"""
print("======================================")
print("Now is the Tess Light Curve.")
print("======================================")
print("Load and process data source1")
val_data_path = r"raw_data/tess/valid.pt"
val_lc, val_label, batch_res = build_patch_data_tess(val_data_path, save_dir = "dataset_t/val")

np_array = np.array(batch_res)
print("batch_res: ", np_array.shape)
print("sum: ", sum(batch_res))
np.save('dataset_t12/patch_ids_source1.npy', np_array)

print("======================================")
print("Data source1 X's shape is ", val_lc.shape)
print("Data source1 y's shape is ", val_label.shape)
print("Shape is: (patch_num, dim, d_model)")

print("======================================")
print("Load and process data source2")
test_data_path = r"raw_data/tess/test.pt"
test_lc, test_label, batch_res = build_patch_data_tess(test_data_path, save_dir="dataset_t/test")

np_array = np.array(batch_res)
print("batch_res: ", np_array.shape)
print("sum: ", sum(batch_res))
np.save('dataset_t12/patch_ids_source2.npy', np_array)

print("======================================")
print("Data source2 X's shape is ", test_lc.shape)
print("Data source2 y's shape is ", test_label.shape)
print("Shape is: (patch_num, dim, d_model)")

print("======================================")
print("Build test data from source 1 & 2 data")

# Follow my exp choose data 策略 TODO
test_lc, val_lc = val_lc, test_lc
test_label, val_label = val_label, test_label


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
save_as_numpy(data_lc_test, data_label_test, save_dir="dataset_t/test")

print("Build val data from source 1 & 2 data")
data_lc_val = val_lc[:val_num]
data_label_val = val_label[:val_num]
save_as_numpy(data_lc_val, data_label_val, save_dir="dataset_t/val")

print("Build train data from source 1 & 2 data")
left_lc_test = test_lc[test_num:]
left_label_test = test_label[test_num:]
left_lc_val = val_lc[val_num:]
left_label_val = val_label[val_num:]

train_lc = np.vstack([left_lc_val, left_lc_test])  # Overlay
train_label = np.vstack([left_label_val, left_label_test])
train_lc = torch.from_numpy(train_lc)
train_label = torch.from_numpy(train_label)
save_as_numpy(train_lc, train_label, save_dir="dataset_t/train")
print("Train data X's shape is ", train_lc.shape)
print("Train data y's shape is ", train_label.shape)
print("Test data X's shape is ", data_lc_test.shape)
print("Test data y's shape is ", data_label_test.shape)
print("Val data X's shape is ", data_lc_val.shape)
print("Val data y's shape is ", data_label_val.shape)
print("Shape is: (patch_num, dim, d_model)")
print("Tess dataset which saved in dataset_t built over!")
print("======================================")


print("======================================")
print("Balance data....................................")
test_lc_path = 'dataset_t/test/lc_data.npy'
test_label_path = 'dataset_t/test/label_data.npy'

val_lc_path = 'dataset_t/val/lc_data.npy'
val_label_path = 'dataset_t/val/label_data.npy'

train_lc_path = 'dataset_t/train/lc_data.npy'
train_label_path = 'dataset_t/train/label_data.npy'

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



from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch_optimizer as optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import auc as aupr
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_auc_score

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data TODO.这里需要改.
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        # self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len) 注.tyx. 我这里都是1000
        self.args.seq_len = 512
        self.args.pred_len = 0 # todo 这里是0感觉不对，调试一下 ,分类任务中用不到，正确
        self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names) # 注.我这里有2类 tyx.
        self.args.num_class = 2
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss() # TODO 这里需要更改
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                # loss = criterion(pred, label.long().squeeze().cpu())
                loss = criterion(pred, label.squeeze(dim=1).long().cpu()) # TODO 这里修改了z
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        f1 = f1_score(trues, predictions, average='weighted')
        print("验证集f1分数为: ", f1)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN') # TODO,其中的代码，较难分析 ？
        vali_data, vali_loader = self._get_data(flag='VAL') # todo
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path) # TODO. label 创建目录

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer() # TODO. 优化算法选择Adam
        criterion = self._select_criterion() # TODO. loss函数选择交叉熵损失函数

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time() #

            for i, (batch_x, label, padding_mask) in enumerate(train_loader): # TODO. false.到这里又出错了，num_workers改成0没问题了
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device) # batch_x. (16, 152, 3)
                padding_mask = padding_mask.float().to(self.device) # padding_mask. (16, 152)
                label = label.to(self.device) # label. (16, 1)

                outputs = self.model(batch_x, padding_mask, None, None) # 回来分析这里，看看这里有没有加上MLP模型用来做分类任务 TODO shape.(16, 26) 主要看看pred_len这个东东有没有发挥作用 -> 没有
                loss = criterion(outputs, label.long().squeeze(-1)) # outputs. (16, 26) .. (16)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()

        # 注. tyx. 补充... ↓
        # 计算指标
        f1 = f1_score(trues, predictions, average='weighted')
        recall = recall_score(trues, predictions, average='weighted')
        precision = precision_score(trues, predictions, average='weighted')

        probs_cpu = probs.cpu().detach().numpy()
        precision_au, recall_au, _ = precision_recall_curve(trues, probs_cpu[:, 1])
        auprc = aupr(recall_au, precision_au)

        auc = roc_auc_score(trues, probs_cpu[:, 1])
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 打印指标
        print("测试集精度如下:. ..")
        print('accuracy:{}'.format(accuracy))
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'AUC: {auc:.4f}')
        print(f"AUPRC: {auprc:.4f}")

        # 注. tyx. 补充...↑

        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return

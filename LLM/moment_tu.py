import argparse
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import AttributeDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from momentfm import MOMENTPipeline

# ---------------------
# 1. 定义本地模型路径（moment）
# ---------------------
LOCAL_MODEL_PATH = "./moment_base"  # 修改为moment本地路径 -> moment-1-large


# ---------------------
# 2. 数据集类
# ---------------------
def load_kepler_data(path):
    """加载光变曲线和标签数据"""
    lc_data = torch.from_numpy(np.load(os.path.join(path, "lc_data.npy")).astype(np.float32))
    label_data = torch.from_numpy(np.load(os.path.join(path, "label_data.npy")).astype(np.int64))

    # 展平数据以适应模型输入
    X = lc_data
    X_flattened = X  # X shape is [B, c_in, len], right
    y = label_data
    y_flattened = y.view(y.size(0))

    # todo
    # X_flattened = X_flattened[0:100]
    # y_flattened = y_flattened[0:100]
    return X_flattened, y_flattened


class CustomDataset(Dataset):
    def __init__(self, path):
        self.inputs, self.labels = load_kepler_data(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 直接返回 (输入张量, 标签张量)
        return self.inputs[idx], self.labels[idx]


# ---------------------
# 3. 数据模块
# ---------------------
class CustomDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, test_path, batch_size=16, num_workers=10):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(self.train_path)
        self.val_dataset = CustomDataset(self.val_path)
        self.test_dataset = CustomDataset(self.test_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# ---------------------
# 4. 自定义模型 此处不需要
# ---------------------


# ---------------------
# 5. LightningModule封装
# ---------------------
class MomentLightningModule(LightningModule):
    def __init__(self, num_classes=2, input_dim=1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 初始化模型
        self.model = MOMENTPipeline.from_pretrained(LOCAL_MODEL_PATH, model_kwargs={
            'task_name': 'classification',
            'n_channels': 1,  # todo 此处应该也会用到修改.
            'num_class': 2
        })
        self.model.init()
        # print("模型如下: ")  # todo 调试一下看看模型的相关属性是否加载.
        # print(self.model)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x_enc):
        outputs = self.model(x_enc=x_enc)
        return outputs


    @classmethod
    def load_from_saved_model(cls, path, **kwargs):
        """从保存的模型加载"""
        # 加载配置
        config_path = os.path.join(path, "config.bin")
        if os.path.exists(config_path):
            add_safe_globals([AttributeDict])

            config = torch.load(config_path)
            # 合并用户提供的参数和保存的配置
            for key, value in config.items():
                if key not in kwargs:
                    kwargs[key] = value

        # 创建模型实例
        model = cls(**kwargs)

        # 加载模型权重
        model_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            # 处理可能的模块前缀
            model_to_load = model.model.module if hasattr(model.model, 'module') else model.model
            model_to_load.load_state_dict(state_dict)

        # 加载LightningModule状态
        lightning_module_path = os.path.join(path, "lightning_module.bin")
        if os.path.exists(lightning_module_path):
            add_safe_globals([AttributeDict])

            checkpoint = torch.load(lightning_module_path,weights_only=False)
            # 只加载需要的状态，避免覆盖已加载的模型权重
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        return model

    def training_step(self, batch, batch_idx):
        # todo
        # print(f"批处理输出类型: {type(batch)}")  # 应输出 <class 'tuple'>
        # print(f"输入形状: {batch[0].shape}")  # 应输出 [B, n_channels, forecast_horizon]
        # print(f"标签形状: {batch[1].shape}")  # 应输出 [B]
        x, y = batch
        outputs = self(x_enc = x)
        loss = self.criterion(outputs.logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x_enc = x)
        logits = outputs.logits

        loss = self.criterion(logits, y)

        # 计算预测
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        y_true = y.cpu().numpy()

        # 计算预测
        # 存储结果用于epoch结束计算
        self.validation_outputs.append({
            'loss': loss,
            'preds': preds.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'y_true': y_true
        })
        return loss

    def on_validation_epoch_end(self):
        all_preds = np.concatenate([o['preds'] for o in self.validation_outputs])
        all_y_true = np.concatenate([o['y_true'] for o in self.validation_outputs])
        all_probs = np.concatenate([o['probs'] for o in self.validation_outputs])
        avg_loss = np.mean([o['loss'].item() for o in self.validation_outputs])

        # 计算全局指标
        self.log('val_loss', avg_loss, sync_dist=True)
        self.log('val_accuracy', accuracy_score(all_y_true, all_preds), sync_dist=True)
        self.log('val_precision', precision_score(all_y_true, all_preds, average='weighted'), sync_dist=True)
        self.log('val_recall', recall_score(all_y_true, all_preds, average='weighted'), sync_dist=True)
        self.log('val_f1', f1_score(all_y_true, all_preds, average='weighted'), sync_dist=True)

        # 对于二分类添加额外指标
        if len(np.unique(all_y_true)) == 2:
            self.log('val_auc', roc_auc_score(all_y_true, all_probs[:, 1]), sync_dist=True)
            self.log('val_auprc', average_precision_score(all_y_true, all_probs[:, 1]), sync_dist=True)

        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x_enc=x)
        logits = outputs.logits
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        y_true = y.cpu().numpy()

        # 计算预测
        # 存储结果用于epoch结束计算
        self.test_outputs.append({
            'loss': loss,
            'preds': preds.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'y_true': y_true
        })
        return loss


    def on_test_epoch_end(self):
        # 复用验证结束逻辑但使用test前缀
        all_preds = np.concatenate([o['preds'] for o in self.test_outputs])
        all_y_true = np.concatenate([o['y_true'] for o in self.test_outputs])
        all_probs = np.concatenate([o['probs'] for o in self.test_outputs])
        avg_loss = np.mean([o['loss'].item() for o in self.test_outputs])

        self.log('test_loss', avg_loss,sync_dist=True)
        self.log('test_accuracy', accuracy_score(all_y_true, all_preds),sync_dist=True)
        self.log('test_precision', precision_score(all_y_true, all_preds, average='weighted'),sync_dist=True)
        self.log('test_recall', recall_score(all_y_true, all_preds, average='weighted'),sync_dist=True)
        self.log('test_f1', f1_score(all_y_true, all_preds, average='weighted'),sync_dist=True)
        self.log('test_auc', roc_auc_score(all_y_true, all_probs[:, 1]),sync_dist=True)
        self.log('test_auprc', average_precision_score(all_y_true, all_probs[:, 1]),sync_dist=True)

        self.test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    def save_model(self, path):
        """保存模型，兼容PyTorch Lightning的检查点格式"""
        os.makedirs(path, exist_ok=True)

        # 保存模型权重
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))

        # 保存配置信息
        config = {
            'num_classes': self.hparams.num_classes,
            'input_dim': self.hparams.input_dim,
            'lr': self.hparams.lr,
            # 添加其他需要的配置参数
        }
        torch.save(config, os.path.join(path, "config.bin"))

        # 保存完整的LightningModule状态
        torch.save({
            'state_dict': self.state_dict(),
            'hparams': self.hparams,
            # 可以添加其他需要保存的状态
        }, os.path.join(path, "lightning_module.bin"))

        print(f"Model saved to {path}")

def main(args):
    # 初始化数据模块
    data_module = CustomDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 初始化LightningModule
    model = MomentLightningModule(
        num_classes=args.num_classes,
        input_dim=args.input_dim,
        lr=args.lr
    )

    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=args.output_dir,
        filename='moment-best-model',
        save_top_k=1,
        mode='max'
    )

    # 配置早停回调（耐心值10轮）
    early_stopping = EarlyStopping(
        monitor='val_f1',  # 监视验证准确率
        patience=10,  # 早停轮数
        mode='max',  # 最大化准确率
        verbose=True,
        check_finite=True
    )

    # 配置TensorBoard日志
    logger = TensorBoardLogger(save_dir='logs', name='moment')

    # 初始化Trainer，添加早停回调
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.gpus,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True
    )

    # 训练模型
    trainer.fit(model, data_module)

    # 保存最终模型（如果未被早停）
    model.save_model(args.output_dir)

    # ---------------------
    # 加载最佳模型并进行测试
    # ---------------------
    print("Loading best model for testing...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        # 在加载模型前添加安全全局对象
        add_safe_globals([AttributeDict])

        # 从检查点加载模型
        model = MomentLightningModule.load_from_saved_model(
            args.output_dir,
        )

        # 运行测试
        print("Running test on test dataset...")
        trainer.test(model, data_module.test_dataloader())

    else:
        print("No best model found, using last trained model for testing")
        # 使用最后训练的模型进行测试
        trainer.test(model, data_module.test_dataloader())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Moment using DataParallel')

    # 数据参数
    parser.add_argument('--train_path', type=str, default='./dataset/train', help='Path to training data')
    parser.add_argument('--test_path', type=str, default='./dataset/test', help='Path to test data')
    parser.add_argument('--val_path', type=str, default='./dataset/val', help='Path to val data')
    parser.add_argument('--output_dir', type=str, default='./moment_saved', help='Output directory for saved model')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs to use (-1 for all available)')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行主函数
    main(args)



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from einops import rearrange
from lightning_fabric.utilities import AttributeDict

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score
from torch.serialization import add_safe_globals
from transformers import GPT2Model




from tools import DataEmbedding
from modules import CustomDataModule

# ---------------------
# 1. 定义本地模型路径 # 注. 这个需要本地模型gpt2
# ---------------------
LOCAL_MODEL_PATH = "./gpt2_local"


class gpt4ts(nn.Module):
    def __init__(self):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = 512
        self.max_len = 512
        self.patch_size = 16
        self.stride = 2
        self.gpt_layers = 6
        self.feat_dim = 2  # todo
        self.num_classes = 2
        self.d_model = 768

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, 768, 0.1)

        self.gpt2 = GPT2Model.from_pretrained("./gpt2_local", output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(768 * self.patch_num)

        self.ln_proj = nn.LayerNorm(768 * self.patch_num)
        self.out_layer = nn.Linear(768 * self.patch_num, self.num_classes)

    def forward(self, x_enc):
        # B, L, M = x_enc.shape
        #
        # input_x = rearrange(x_enc, 'b l m -> b m l')

        # input = x_enc.unsqueeze(1) # todo
        input = x_enc.view(x_enc.size(0), 2, 512)  # 若数据是2维

        B, M, L = input.shape

        input_x = self.padding_patch_layer(input)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        outputs = self.enc_embedding(input_x, None)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs


# ---------------------
# 4. 自定义模型
# ---------------------
class CustomModel(nn.Module):
    def __init__(self, num_classes, input_dim=1):
        super().__init__()
        self.gpt4ts = gpt4ts()

    def forward(self, x_enc=None, **kwargs):
        logits = self.gpt4ts(x_enc)
        return logits


# ---------------------
# 5. LightningModule封装
# ---------------------
class Gpt4tsLightningModule(LightningModule):
    def __init__(self, num_classes=2, input_dim=1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 初始化模型
        self.model = CustomModel(num_classes, input_dim)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, enc):
        return self.model(enc)

    @classmethod
    def load_from_saved_model(cls, path, **kwargs):
        """从保存的模型加载"""
        # 加载配置
        config_path = os.path.join(path, "config.bin")
        if os.path.exists(config_path):
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
            checkpoint = torch.load(lightning_module_path)
            # 只加载需要的状态，避免覆盖已加载的模型权重
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        return model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(enc=x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(enc=x)
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
        logits = self(enc=x)
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

        self.log('test_loss', avg_loss, sync_dist=True)
        self.log('test_accuracy', accuracy_score(all_y_true, all_preds), sync_dist=True)
        self.log('test_precision', precision_score(all_y_true, all_preds, average='weighted'), sync_dist=True)
        self.log('test_recall', recall_score(all_y_true, all_preds, average='weighted'), sync_dist=True)
        self.log('test_f1', f1_score(all_y_true, all_preds, average='weighted'), sync_dist=True)
        self.log('test_auc', roc_auc_score(all_y_true, all_probs[:, 1]), sync_dist=True)
        self.log('test_auprc', average_precision_score(all_y_true, all_probs[:, 1]), sync_dist=True)

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



# ---------------------
# 6. 主函数（添加早停）
# ---------------------
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
    model = Gpt4tsLightningModule(
        num_classes=args.num_classes,
        input_dim=args.input_dim,
        lr=args.lr
    )

    # 配置早停回调（耐心值10轮）
    early_stopping = EarlyStopping(
        monitor='val_f1',  # 监视验证准确率
        patience=10,  # 早停轮数
        mode='max',  # 最大化准确率
        verbose=True,
        check_finite=True
    )

    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=args.output_dir,
        filename='gpt4ts-best-model',
        save_top_k=1,
        mode='max'
    )


    # 配置TensorBoard日志
    logger = TensorBoardLogger(save_dir='logs', name='lora-gpt4ts')

    # 初始化Trainer，添加早停回调
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.gpus,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        strategy=DDPStrategy(find_unused_parameters=True)
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
        model = Gpt4tsLightningModule.load_from_saved_model(
            args.output_dir,
        )

        # 运行测试
        print("Running test on test dataset...")
        trainer.test(model, data_module.test_dataloader())

    else:
        print("No best model found, using last trained model for testing")
        # 使用最后训练的模型进行测试
        trainer.test(model, data_module.test_dataloader())
        final_model_path = os.path.join(args.output_dir, 'final_test_model')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA fine-tuning with gpt2 using PyTorch Lightning')

    # 数据参数
    parser.add_argument('--train_path', type=str, default='./dataset12/train', help='Path to training data')
    parser.add_argument('--test_path', type=str, default='./dataset12/test', help='Path to test data')
    parser.add_argument('--val_path', type=str, default='./dataset12/val', help='Path to val data')
    parser.add_argument('--output_dir', type=str, default='./gpt4ts_saved_1_12', help='Output directory for saved model')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--input_dim', type=int, default=2, help='Input feature dimension') # todo

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (早停可能提前终止)')  # todo
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs to use (-1 for all available)')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行主函数
    main(args)

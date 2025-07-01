# z注.这是 GPT-2 的最小版本，包含 12400 万个参数。
# **相关模型**：GPT-Large、GPT-Medium 和 GPT-XL
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import  LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import GPT2Model
from peft import get_peft_model, LoraConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score

import numpy as np
import argparse
from modules import CustomDataModule


# ---------------------
# 1. 定义本地模型路径（gpt2）
# ---------------------
LOCAL_MODEL_PATH = "./gpt2_local"


# ---------------------
# 4. 自定义模型
# ---------------------
class Adapter(nn.Module):
    """小型Adapter层，插入到每个Transformer模块之后"""

    def __init__(self, input_dim, reduction_factor=16):
        super().__init__()
        self.down_project = nn.Linear(input_dim, input_dim // reduction_factor)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(input_dim // reduction_factor, input_dim)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x  # 残差连接


class CustomModel(nn.Module):
    def __init__(self, num_classes, input_dim=2):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")  # 实际使用时替换为LOCAL_MODEL_PATH
        self.config = self.gpt2.config

        # 冻结GPT2基础参数
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # 输入投影层和分类头
        self.input_proj = nn.Linear(input_dim, self.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.add_adapters()

    def add_adapters(self):
        """在GPT2的每个encoder层后添加Adapter"""
        for i in range(len(self.gpt2.h)):
            # 为注意力输出添加Adapter
            adapter_attn = Adapter(self.config.hidden_size)
            setattr(self, f'adapter_attn_{i}', adapter_attn)

            # 为前馈网络输出添加Adapter
            adapter_ffn = Adapter(self.config.hidden_size)
            setattr(self, f'adapter_ffn_{i}', adapter_ffn)

    def forward(self, input_ids=None, **kwargs):
        # 扩展维度并投影到GPT2嵌入空间
        x = input_ids.view(input_ids.size(0), 512, 2)  # 假设输入数据是2维
        embedded = self.input_proj(x)

        # 应用GPT2的embedding层
        hidden_states = self.gpt2.wte(input_ids) + self.gpt2.wpe(torch.arange(embedded.size(1), device=embedded.device))
        hidden_states = self.gpt2.drop(hidden_states + embedded)

        # 获取注意力掩码
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=self.gpt2.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # 依次通过每个transformer层和对应的Adapter
        for i in range(len(self.gpt2.h)):
            layer = self.gpt2.h[i]

            # 应用层归一化和注意力子层
            residual = hidden_states
            hidden_states = layer.ln_1(hidden_states)
            attn_outputs = layer.attn(
                hidden_states,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False,
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            hidden_states = residual + attn_output

            # 应用注意力后的Adapter
            hidden_states = getattr(self, f'adapter_attn_{i}')(hidden_states)

            # 应用层归一化和MLP子层
            residual = hidden_states
            hidden_states = layer.ln_2(hidden_states)
            feed_forward_hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states

            # 应用前馈网络后的Adapter
            hidden_states = getattr(self, f'adapter_ffn_{i}')(hidden_states)

        # 应用最终层归一化
        hidden_states = self.gpt2.ln_f(hidden_states)

        # 获取最终的CLS嵌入并通过分类头
        cls_embedding = hidden_states[:, 0, :]  # 使用第一个token作为分类标记
        logits = self.classifier(cls_embedding)
        return logits

    # @staticmethod # todo加上这个了，训练参数是：492k, 去掉这个：训练参数是：492也没变
    def setup_training_params(model):
        """设置训练参数，只解冻Adapter和分类头"""
        for name, param in model.named_parameters():
            # if 'adapter' in name or 'classifier' in name or 'input_proj' in name: # todo加上这个了，训练参数是：492k,
            #     param.requires_grad = True
            # else:
            if 'lora_A' in name or 'lora_B' in name or 'lora_a' in name or 'lora_b' in name or'adapter' in name or 'classifier' in name or 'input_proj' in name:
                param.requires_grad = True # 如果用这个，训练参数是：？没变也是 492k
            else:
                param.requires_grad = False
        return model

class Adapter(nn.Module):
    """小型Adapter层，插入到每个Transformer模块之后"""

    def __init__(self, input_dim, reduction_factor=16):
        super().__init__()
        self.down_project = nn.Linear(input_dim, input_dim // reduction_factor)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(input_dim // reduction_factor, input_dim)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x  # 残差连接

class CustomModel(torch.nn.Module):
    def __init__(self, num_classes, input_dim=2): # todo
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(LOCAL_MODEL_PATH)
        self.config = self.gpt2.config

        # 冻结gpt2基础参数
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # 入投影层和分类头
        self.input_proj = torch.nn.Linear(input_dim, self.config.hidden_size)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.gpt2.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )


    def forward(self, input_ids, **kwargs):
        # x = input_ids.unsqueeze(-1)
        x = input_ids.view(input_ids.size(0), 512, 2)  # 若数据是2维 todo
        embedded = self.input_proj(x)
        outputs = self.gpt2(inputs_embeds=embedded)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


# ---------------------
# 5. LightningModule封装
# ---------------------
class LoraBertLightningModule(LightningModule):
    def __init__(self, num_classes=2, input_dim=1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 初始化模型
        self.model = CustomModel(num_classes, input_dim)
        print(self.model)  # todo

        # 配置LoRA
        self.peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            # target_modules=["attn.c_attn"],
            target_modules=["attn.c_attn"], #  todo ,增加了一个微调注意力相关的线性层,即整个自注意力子层
            # idea3. 注.先不微调最后一个ln层
            # target_modules=[
            #     # 注意力模块的线性层（保持原有配置）
            #     "attn.c_attn",
            #     "attn.c_proj",
            #     # 注意力输出后的LayerNorm（ln_1）
            #     "h.*.ln_1",
            #     # 全连接层输出后的LayerNorm（ln_2）
            #     "h.*.ln_2",
            # ],
            bias="none",
            modules_to_save=["classifier"]
        )

        # 应用LoRA
        self.lora_model = get_peft_model(self.model, self.peft_config)
        self.lora_model.print_trainable_parameters()

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, input_ids):
        return self.lora_model(input_ids)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(input_ids=x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(input_ids=x)
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
        logits = self(input_ids=x)
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
        return torch.optim.AdamW(self.lora_model.parameters(), lr=self.hparams.lr)

    def save_model(self, path):
        model_to_save = self.lora_model.module if hasattr(self.lora_model, 'module') else self.lora_model
        model_to_save.save_pretrained(path)
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
    model = LoraBertLightningModule(
        num_classes=args.num_classes,
        input_dim=args.input_dim,
        lr=args.lr
    )

    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=args.output_dir,
        filename='gpt-2-best-model-{epoch:02d}-{val_f1:.4f}',
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
    logger = TensorBoardLogger(save_dir='logs', name='lora-gpt2')

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
    final_model_path = os.path.join(args.output_dir, 'final_model')
    model.save_model(final_model_path)

    # ---------------------
    # 加载最佳模型并进行测试
    # ---------------------
    print("Loading best model for testing...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        # 从检查点加载模型
        model = LoraBertLightningModule.load_from_checkpoint(
            best_model_path,
            num_classes=args.num_classes,
            input_dim=args.input_dim,
            lr=args.lr
        )

        # 运行测试
        print("Running test on test dataset...")
        trainer.test(model, data_module.test_dataloader())

        # 保存测试后的最佳模型
        final_model_path = os.path.join(args.output_dir, 'best_test_model')
        model.save_model(final_model_path)
    else:
        print("No best model found, using last trained model for testing")
        # 使用最后训练的模型进行测试
        trainer.test(model, data_module.test_dataloader())
        final_model_path = os.path.join(args.output_dir, 'final_test_model')
        model.save_model(final_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA fine-tuning with GPT2 using DataParallel')
    parser.add_argument('--train_path', type=str, default='./dataset12_k/train', help='Path to training data') #todo
    parser.add_argument('--val_path', type=str, default='./dataset12_k/val', help='Path to training data')
    parser.add_argument('--test_path', type=str, default='./dataset12_k/test', help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./gpt2_saved12_l', help='Output directory for saved model')

    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--input_dim', type=int, default=2, help='Input feature dimension') # todo

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs') # todo
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs to use (-1 for all available)')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)

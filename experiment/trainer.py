import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from tqdm import tqdm
import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score
import json
import pytz
import time
import numpy as np
from gradnorm_pytorch import GradNormLossWeighter
from model.BGC_models import BGCClassifier, ProductMatching

def initialize(config):
    model_parameters = config.model_parameters
    if config.task == "classification":
        criterion = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor(i),reduction="sum") for i in config.BCEWithLogitsLoss.pos_weights]
        model = BGCClassifier(esm_size = model_parameters.esm_size,
                                        num_classes = model_parameters.num_classes,
                                        attention_dim = model_parameters.attention_dim,
                                        num_heads = model_parameters.num_heads,
                                        dropout = model_parameters.dropout)
    elif config.task == "product_matching":
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(config.BCEWithLogitsLoss.pos_weights), reduction="sum")
        model = ProductMatching(esm_size = model_parameters.esm_size,
                                   gearnet_size = model_parameters.gearnet_size,
                                   attention_dim = model_parameters.attention_dim,
                                   num_heads = model_parameters.num_heads,
                                   vocab_size = model_parameters.vocab_size,
                                   dropout = model_parameters.dropout)
    else:
        raise ValueError(f"Unknown task: {config.task}")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.optimizer.lr), betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model = model.to(config.device)
    return criterion, model, optimizer, scheduler


def move_to_device(config_or_device, *args):
    if isinstance(config_or_device, (torch.device, str)):
        device = torch.device(config_or_device) if isinstance(config_or_device, str) else config_or_device
    
    elif hasattr(config_or_device, 'device'):
        device = config_or_device.device
    
    else:
        raise ValueError(
            "Input must be either: "
            "(1) a torch.device object, "
            "(2) a device string (e.g., 'cuda'), "
            "or (3) a config object with a 'device' attribute."
        )
    
    return [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]

def accuracy_cal(true: torch.Tensor, pred: torch.Tensor, threshold: float) -> np.array:
  '''
  true: Tensor, (len(dataloader), num_classes)
  pred: Tnesor, (len(dataloader), num_classes)
  return: Tnesor, (num_classes,)
  '''
  data_size = true.shape[0]
  pred_labels = (pred > threshold).float()
  correct_pred = torch.sum((true == pred_labels), dim=0)
  accuracy_per_class = correct_pred / data_size
  return accuracy_per_class.cpu().numpy()

def precision_cal(true:torch.Tensor, pred:torch.Tensor, threshold) -> np.array:
    '''
    true: Tensor, (len(dataloader), num_classes)
    pred: Tensor, (len(dataloader), num_classes)
    '''
    pred_labels = (pred > threshold).float()
    true_positive = torch.sum((true == 1) & (pred_labels == 1), dim=0)
    false_positive = torch.sum((true == 0) & (pred_labels == 1), dim=0)

    precision_per_class = true_positive / (true_positive + false_positive + 1e-10)
    return precision_per_class.cpu().numpy()

def recall_cal(true:torch.Tensor, pred:torch.Tensor, threshold) -> np.array:
    '''
    true: Tensor, (len(dataloader), num_classes)
    pred: Tensor, (len(dataloader), num_classes)
    '''
    pred_labels = (pred > threshold).float()
    true_positive = torch.sum((true == 1) & (pred_labels == 1), dim=0)
    false_negative = torch.sum((true == 1) & (pred_labels == 0), dim=0)

    recall_per_class = true_positive / (true_positive + false_negative + 1e-10)
    return recall_per_class.cpu().numpy()

def f1_cal(recall:np.array, precision:np.array, mode="micro") -> np.array:
    '''
    recall: np.array, (num_classes,)
    precision: np.array, (num_classes,)
    mode: str, either "micro" or "macro"
    return:
      f1_score: np.array (1,)
      f1_per_class: np.array (num_classes,)
    '''
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-10)
    if mode == "micro":
      precision_mean = np.mean(precision)
      recall_mean = np.mean(recall)
      f1_score = 2 * precision_mean * recall_mean / (precision_mean + recall_mean + 1e-10)
    elif mode == "macro":
      f1_score = np.mean(f1_per_class)
    else:
      raise ValueError("Mode must be either 'micro' or 'macro'.")
    return f1_score, f1_per_class

def auc_cal(true, pred, per_class=True) -> np.array:
    '''
    true: Tensor, (len(dataloader), num_classes)
    pred: Tensor, (len(dataloader), num_classes)
    per_class: bool
    return:
      auc: np.array
    '''
    true_np = true.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    if per_class:
        auc_per_class = []
        for i in range(true_np.shape[1]):
            auc = roc_auc_score(true_np[:, i], pred_np[:, i])
            auc_per_class.append(auc)
        return np.array(auc_per_class)
    else:
        auc = roc_auc_score(true_np.ravel(), pred_np.ravel()) #flatten the numpy array
        return auc

def metrics_text(class_dict, metrics: dict, show_metric):
    metrics_summary = ""
    for index, biosyn_class in enumerate(class_dict.keys()):
        metrics_summary += f"{biosyn_class}: "
        for metric in ["accuracy_per_class", "recall_per_class", "precision_per_class", "f1_per_class", "auc_per_class"]:
            if metric in show_metric:
                metric_name = metric.split("_")[0]
                metrics_summary += f"{metric_name}: {metrics[metric][0][index]:.4f} "
        metrics_summary += "\n"

    for metric in ["overall_accuracy", "overall_recall", "overall_precision", "micro_f1", "macro_f1", "overall_auc"]:
        if metric in show_metric:
            metrics_summary += f"{metric}: {metrics[metric][0]:.4f} "
            metrics_summary += "\n"

    return metrics_summary

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, metrics, config, verbose):
        self.config = config
        self.verbose = verbose
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = config.device
        self.epochs = config.epochs
        self.metrics = metrics
        self.train_metrics = {}
        self.val_metrics = {}
        self.threshold = 0.5
        self.writer = SummaryWriter(log_dir=os.path.join("runs", "tensorboard", self._get_tensorboard_dir()))
        self.patience = config.patience
        
        if self.verbose:
            self._setup_logging(config.log_dir)
    
    def _get_tensorboard_dir(self):
        raise NotImplementedError
    
    def _setup_logging(self, log_dir):
        tz = pytz.timezone('Asia/Shanghai')
        log_filename = f"{self._get_log_prefix()} {datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_file_path = os.path.join(log_dir, log_filename)
        print("Log file will be saved at:", log_file_path)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            for handler in self.logger.handlers:
                handler.flush()
                handler.close()
            self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if self.verbose == 2:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
    
    def _get_log_prefix(self):
        raise NotImplementedError
    
    def calculate_metrics(self, true, pred):
        metrics = {}
        recall_per_class = recall_cal(true, pred, self.threshold)
        precision_per_class = precision_cal(true, pred, self.threshold)
        if 'accuracy' in self.metrics:
            accuracy_per_class = accuracy_cal(true, pred, self.threshold)
            metrics['accuracy_per_class'] = [accuracy_per_class]
            metrics["overall_accuracy"] = [np.mean(accuracy_per_class)]
        if 'recall' in self.metrics:
            metrics['recall_per_class'] = [recall_per_class]
            metrics["overall_recall"] = [np.mean(recall_per_class)]
        if 'precision' in self.metrics:
            metrics['precision_per_class'] = [precision_per_class]
            metrics["overall_precision"] = [np.mean(precision_per_class)]
        if 'auc' in self.metrics:
            metrics['auc_per_class'] = [auc_cal(true, pred, per_class=True)]
            metrics['overall_auc'] = [auc_cal(true, pred, per_class=False)]
        if "micro_f1" in self.metrics:
            micro_f1 = f1_cal(recall_per_class, precision_per_class, mode="micro")
            metrics['f1_per_class'] = [micro_f1[1]]
            metrics['micro_f1'] = [micro_f1[0]]
        if "macro_f1" in self.metrics:
            macro_f1 = f1_cal(recall_per_class, precision_per_class, mode="macro")
            metrics['f1_per_class'] = [macro_f1[1]]
            metrics['macro_f1'] = [macro_f1[0]]
        return metrics
    
    def train(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
    
    def run(self, overall_monitor=None, class_monitor=None, show_progress=False):
        best_val_loss = float("inf")
        start_time = time.time()
        best_pred = torch.empty(0)
        no_improve_count = 0
        
        for epoch in tqdm(range(self.epochs), desc='Epochs', disable=not show_progress):
            train_metrics = self.train()
            val_metrics, pred = self.validate()
            #print("epoch:",epoch, "validation metrics",val_metrics)
            train_loss, validation_loss = train_metrics["avg_train_loss"], val_metrics["avg_val_loss"]
            
            self.writer.add_scalar('loss/train_loss', train_loss, epoch)
            self.writer.add_scalar('loss/validation_loss', validation_loss, epoch)
            
            if val_metrics["avg_val_loss"] < best_val_loss:
                best_val_loss = val_metrics["avg_val_loss"]
                best_pred = pred
                best_model = self.model
                best_epoch = epoch
                best_metrics = val_metrics
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if self.verbose:
                self.logger.info("User data: %s", json.dumps(self.config.to_dict(), indent=4))
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}\n Training Loss: {train_loss}, validation_loss:{validation_loss}")
                self.logger.info(metrics_text(self.class_dict, val_metrics, list(val_metrics.keys())))
            
            if overall_monitor:
                for metric in overall_monitor:
                    self.writer.add_scalar(f'{metric}', val_metrics[metric][0].item(), epoch)
            
            if class_monitor:
                for metric in class_monitor:
                    for biosyn_class in self.class_dict:
                        class_index = self.class_dict[biosyn_class]
                        self.writer.add_scalar(f'{metric}/{biosyn_class}', val_metrics[metric][0][class_index].item(), epoch)
            
            if no_improve_count >= self.patience:
                break
        
        elapsed_time = time.time() - start_time
        metric_summary = metrics_text(self.class_dict, best_metrics, list(val_metrics.keys()))
        
        if self.verbose:
            self.logger.info(f"——————TOTAL_TIME:{elapsed_time}s,BEST_EPOCH:{best_epoch}——————")
            self.logger.info(metric_summary)
            self.writer.close()
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        return best_model, best_metrics, metric_summary, best_epoch, best_pred


class MACTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, metrics, config, verbose):
        super().__init__(model, train_loader, val_loader, optimizer, scheduler, criterion, metrics, config, verbose)
        self.num_classes = config.model_parameters.num_classes
        self.class_dict = { 'NRP': 0, 'Other': 1, 'Polyketide': 2, 'RiPPs': 3, 'Saccharide': 4, 'Terpene': 5}
        fc_weight = self.model.fc.weight
        self.freeze = config.GradNorm.freeze
        if not self.freeze:
            self.grad_norm_weighter = GradNormLossWeighter(
                num_losses=self.num_classes,
                learning_rate=float(config.GradNorm.lr),
                restoring_force_alpha=1,
                grad_norm_parameters=fc_weight
            )
        else:
            self.grad_norm_weighter = GradNormLossWeighter(
                loss_weights=[1.1, .95, 1.1, .85, 1.15, .78],
                frozen=True
            )
    
    def _get_tensorboard_dir(self):
        return "classifier"
    
    def _get_log_prefix(self):
        return "classify"
    
    def train(self):
        metrics = {}
        total_train_loss = 0.0
        origin_train_loss = 0.0
        total_train_loss_per_class = torch.zeros(6, device=self.device)
        total_samples = len(self.train_loader.dataset)
        self.model.train()
        loss_weights = torch.empty((0,6), device=self.device)

        for batch in self.train_loader:  
            total_batch_train_loss = 0.0
            biosyn_class, pro, pro_mask, structure, class_token = move_to_device(self.config, 
                                                                                 batch["biosyn_class"], 
                                                                                 batch["protein_reps_padded"], 
                                                                                 batch["protein_mask"],
                                                                                 batch["structure_padded"],
                                                                                 batch["class_token"])
            output, cross_attn_weights = self.model(pro, class_token, pro_mask, structure)  # output: (batch_size, num_classes)
            self.optimizer.zero_grad()

            batch_train_loss_per_class = torch.zeros(6, device=self.device)
            losses = []
            for i in range(self.num_classes):

                loss = self.criterion[i](output[:, i], biosyn_class[:, i])
                batch_train_loss_per_class[i] = loss
                losses.append(loss)

            losses = torch.stack(losses)  # (7,)
            self.grad_norm_weighter.backward(losses, retain_graph=True)


            weighted_loss = torch.sum(self.grad_norm_weighter.loss_weights.detach() * losses)
            origin_loss = torch.sum(losses)
            weighted_loss.backward()
            loss_weights = torch.cat([loss_weights,self.grad_norm_weighter.loss_weights.unsqueeze(0)], dim=0)

            self.optimizer.step()

            total_train_loss += weighted_loss.item()  
            total_train_loss_per_class += batch_train_loss_per_class.detach() 

        mean_weights = torch.mean(loss_weights, dim=0)
        metrics["avg_train_loss_per_class"] = total_train_loss_per_class / total_samples
        metrics["avg_train_loss"] = total_train_loss / total_samples
        self.scheduler.step()
        return metrics
    
    def validate(self, mean_weights = None):
        if mean_weights is None:
            mean_weights = torch.ones(6, device = self.device)
        metrics={}
        self.model.eval()
        true, pred = torch.empty((0,6), device=self.device), torch.empty((0,6), device=self.device)
        val_loss_per_class = torch.zeros(6,device=self.device)
        with torch.no_grad():
            for batch in self.val_loader: # each batch
                biosyn_class, pro, pro_mask, structure, class_token = move_to_device(self.config, 
                                                                                     batch["biosyn_class"],
                                                                                     batch["protein_reps_padded"],
                                                                                     batch["protein_mask"],
                                                                                     batch["structure_padded"],
                                                                                     batch["class_token"])
                output, cross_attn_weight = self.model(pro, class_token, pro_mask, structure) #output: (batch_size,num_classes)
                true = torch.cat((true, biosyn_class),dim = 0)
                pred = torch.cat((pred, output),dim = 0)

        for i in range(self.num_classes):
            val_loss_per_class[i] = self.criterion[i](pred[:, i], true[:, i]).item() 
        pred = torch.sigmoid(pred) 
        pred = torch.nan_to_num(pred, nan=0.0)
        metrics["avg_val_loss_per_class"] = val_loss_per_class / true.shape[0]
        metrics["avg_val_loss"] = torch.sum(val_loss_per_class*mean_weights) / true.shape[0]
        metrics["origin_val_loss"] =  torch.sum(val_loss_per_class) / true.shape[0]
        metrics["mean_weights"] = mean_weights
        metrics.update(self.calculate_metrics(true, pred))
        return metrics, pred

class MAPTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, metrics, config, verbose):
        super().__init__(model, train_loader, val_loader, optimizer, scheduler, criterion, metrics, config, verbose)
        self.class_dict = {"is_structure":1}
        
    def _get_tensorboard_dir(self):
        return "predictor"
    
    def _get_log_prefix(self):
        return "predict"
    
    def train(self):
        metrics = {}
        train_loss = 0.0
        self.model.train()
        total_samples = len(self.train_loader.dataset)

        for batch in self.train_loader:
            pro, sub, is_product, pro_mask, sub_mask, structure = move_to_device(self.config, 
                                                                                batch["protein_reps_padded"],
                                                                                batch["sub_padded"],
                                                                                batch["is_product"],
                                                                                batch["protein_mask"],
                                                                                batch["sub_mask"],
                                                                                batch["structure_padded"])
            output, cross_attn = self.model(pro, sub, structure, pro_mask, sub_mask)
            self.optimizer.zero_grad()
            output_flat = output.reshape(-1) 
            batch_train_loss = self.criterion(output_flat, is_product)
            batch_train_loss.backward()
            train_loss += batch_train_loss.item()
            self.optimizer.step()

        self.scheduler.step()
        metrics["avg_train_loss"] = train_loss / total_samples
        return metrics
    
    def validate(self):
        val_loss = 0.0
        metrics = {}
        self.model.eval()
        true, pred = torch.empty((0, 1), device=self.device), torch.empty((0, 1), device=self.device)
        
        with torch.no_grad():
            for batch in self.val_loader:
                pro, sub, is_product, pro_mask, sub_mask, structure = move_to_device(self.config, 
                                                                                batch["protein_reps_padded"],
                                                                                batch["sub_padded"],
                                                                                batch["is_product"],
                                                                                batch["protein_mask"],
                                                                                batch["sub_mask"],
                                                                                batch["structure_padded"])
                output, cross_attn = self.model(pro, sub, structure, pro_mask, sub_mask)
                is_product = is_product.unsqueeze(-1)
                true = torch.cat((true, is_product), dim=0)
                pred = torch.cat((pred, output), dim=0)
        
        val_loss = self.criterion(pred, true).item()
        pred = torch.sigmoid(pred)
        metrics["avg_val_loss"] = val_loss / true.shape[0]
        metrics.update(self.calculate_metrics(true, pred))
        
        return metrics, pred

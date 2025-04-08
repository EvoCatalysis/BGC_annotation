from tqdm import tqdm
import torch
import os
import numpy as np
from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from collections import Counter

from trainer import initialize, f1_cal, precision_cal, recall_cal, auc_cal, MACTrainer, MAPTrainer, move_to_device
from model.BGC_models import BGCClassifier, ProductMatching

PROJECT_DIR = Path(__file__).resolve().parent.parent


def move_to_cpu(data: Any) -> Any:
    if hasattr(data, 'to') and callable(data.to) and hasattr(data, 'device'):
        if str(data.device) == 'cpu':
            return data
        try:

            return data.to('cpu')
        except Exception as e:
            print(f"warning: cannot move object of type {type(data)} to CPU: {e}")
            return data 

    elif isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}

    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]

    elif isinstance(data, tuple):
        return tuple(move_to_cpu(item) for item in data)

    else:
        return data

def calculate_metrics(true, pred):
    recall = recall_cal(true, pred, 0.5)
    precision = precision_cal(true, pred, 0.5)
    return {
        "auc_per_class": auc_cal(true, pred, per_class=True),
        "recall_per_class": recall,
        "precision_per_class": precision,
        "f1_per_class": f1_cal(recall, precision, mode="micro")[1]
    }

def kensemble_validation(dataloaders, config, save_checkpoint=True, 
                         checkpoint_name=None, show_progress=True):
    """
    Ensemble validation function for both classifier and structure matching models.
    
    Args:
        dataloaders: List of dataloaders
        config: Model configuration
        model_type: Either "classifier" or "structure_matching"
        save_checkpoint: Whether to save checkpoints
        checkpoint_dir: Directory to save checkpoints
        show_progress: Whether to show progress bars
        current_dir: Current directory (only needed for structure_matching)
    """
    # Set up paths and metrics based on model type
    if config.task == "classification":
        checkpoint_path = os.path.join(PROJECT_DIR, config.checkpoint_dir, checkpoint_name)
        ensemble_metric = {"micro_f1": [], "f1_per_class": [], "overall_auc": [], "auc_per_class": []}
        overall_monitor = ["overall_auc"]
        class_monitor = ["auc_per_class"]
        trainer = MACTrainer
    elif config.task == "product_matching":
        checkpoint_path = os.path.join(PROJECT_DIR, config.checkpoint_dir, checkpoint_name)
        ensemble_metric = {"overall_auc": [], "overall_recall": [], "overall_precision": [], "f1": []}
        overall_monitor = None
        class_monitor = None
        trainer = MAPTrainer
    else:
        raise ValueError(f"Unknown task: {config.task}")
    
    ensemble_model = []
    ensemble_pred = []
    
    for i, dataloader in enumerate(tqdm(dataloaders[1:], desc="ensembles", disable=not show_progress)):
        checkpoint_name = f"{config.task}_{i}.pth"
        train_dataloader = dataloader["train"]
        val_dataloader = dataloader["val"]
        criterion, model, optimizer, scheduler = initialize(config)
        
        classifier = trainer(model=model,
                          train_loader=train_dataloader,
                          val_loader=val_dataloader,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          criterion=criterion,
                          metrics=["accuracy", "recall", "precision", "auc", "micro_f1"],
                          config=config,
                          verbose=0)
        print(f"begin training ensemble{i}")
        best_model, best_metrics, metric_summary, best_epoch, best_pred = classifier.run(
            overall_monitor=overall_monitor, 
            class_monitor=class_monitor, 
            show_progress=show_progress
        )
        
        ensemble_model.append(best_model)
        ensemble_pred.append(best_pred)
        
        if save_checkpoint:
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(best_model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))
        
        # Extract and store metrics based on model type
        if config.task == "classification":
            micro_f1, f1_per_class = best_metrics["micro_f1"], best_metrics["f1_per_class"]
            overall_auc, auc_per_class = best_metrics["overall_auc"], best_metrics["auc_per_class"]
            ensemble_metric["micro_f1"].append(micro_f1)
            ensemble_metric["f1_per_class"].append(f1_per_class)
            ensemble_metric["overall_auc"].append(overall_auc)
            ensemble_metric["auc_per_class"].append(auc_per_class)
        else:  # structure_matching
            micro_f1 = best_metrics["micro_f1"]
            overall_auc = best_metrics["overall_auc"]
            overall_precision = best_metrics["overall_precision"]
            overall_recall = best_metrics["overall_recall"]
            ensemble_metric["f1"].append(micro_f1)
            ensemble_metric["overall_auc"].append(overall_auc)
            ensemble_metric["overall_precision"].append(overall_precision)
            ensemble_metric["overall_recall"].append(overall_recall)
    
    if save_checkpoint:
        torch.save(move_to_cpu(ensemble_pred), os.path.join(checkpoint_path, "ensemble_pred.pth"))
        # Save metrics for classifier model type
        if config.task == "classification":
            torch.save(move_to_cpu(ensemble_metric), os.path.join(checkpoint_path, "ensemble_metric.pth"))
        # Save config for both model types
        OmegaConf.save(config=config, f=os.path.join(checkpoint_path, "ckpt.yaml"))
    
    return ensemble_metric, ensemble_model, ensemble_pred

def attention_analysis(true, pred, gene_kind, pfam, cross_attn_weights, attention_percent = 0.2):
  """
  cross_attn_weights: tensor
  all_pfam: [[], [], [], [], [], [], []]
  """
  cross_attn_weights_np = cross_attn_weights.detach().cpu().numpy()
  all_gene_kind=[[] for i in range(6)]
  all_pfam=[[] for i in range(6)]
  for class_index in range(6):
    for batch_idx in range(cross_attn_weights_np.shape[0]):
      if true[batch_idx,class_index]==1 and pred[batch_idx,class_index]==1:
        class_attention=cross_attn_weights_np[batch_idx,class_index,:]
        proportion=int(attention_percent * len(gene_kind[batch_idx]))
        if proportion>0:
          top_indices=np.argsort(class_attention)[-proportion:]
          non_zero_indices = [idx for idx in top_indices if idx < len(gene_kind[batch_idx])]
          if len(non_zero_indices)>0:
            gene_kind_selected = [gene_kind[batch_idx][idx] for idx in non_zero_indices]
            pfam_selected = [pfam[batch_idx][idx] for idx in non_zero_indices]
            all_gene_kind[class_index].extend(gene_kind_selected)
            all_pfam[class_index].extend(pfam_selected)
  return all_gene_kind, all_pfam

def generate_ensemblelist(dir):
  models = []
  device = "cuda" if torch.cuda.is_available() else "cpu"
  config = OmegaConf.load(os.path.join(dir,"ckpt.yaml"))
  print(config.task)
  for file_name in os.listdir(dir):
    if file_name.endswith(".pth") and config.task in file_name:
        file_path = os.path.join(dir, file_name)
        if config.task == "classification":
            model = BGCClassifier(esm_size = config.model_parameters.esm_size,
                                        num_classes = config.model_parameters.num_classes,
                                        attention_dim = config.model_parameters.attention_dim,
                                        num_heads = config.model_parameters.num_heads,
                                        dropout = config.model_parameters.dropout)
        elif config.task == "product_matching":
            model = ProductMatching(esm_size = config.model_parameters.esm_size,
                                   gearnet_size = config.model_parameters.gearnet_size,
                                   attention_dim = config.model_parameters.attention_dim,
                                   num_heads = config.model_parameters.num_heads,
                                   vocab_size = config.model_parameters.vocab_size,
                                   dropout = config.model_parameters.dropout)
        else:
            raise
        model.load_state_dict(torch.load(file_path, map_location=device))
        models.append(model)
  return models


def kensemble_MACtest(models, test_loader, ensemble_dir = None, mean_result = True):
    device = next(models[0].parameters()).device
    metrics = {"auc_per_class":[], 
               "recall_per_class":[], 
               "precision_per_class":[], 
               "f1_per_class":[]}
    
    with torch.no_grad():
        all_gene_kind=[[] for i in range(6)]
        all_pfam=[[] for i in range(6)]
        gene_kind_high_attn = []
        pfam_high_attn = []
        all_preds = torch.empty((len(models),0,6), device=device)
        all_attn_weight = [[] for i in range(len(models))]
        mean_attn_weight = []
        true = torch.empty((0,6), device=device)
        for batch in tqdm(test_loader, desc="batch"): # each batch
            batch_pred = []
            batch_attn_weight = []
            label, biosyn_class, pro, length, pro_mask, structure, class_token, gene_kind, pfam = move_to_device(device,*batch)
            for i, model in enumerate(models):
                model.eval()
                output, cross_attn_weight = model(pro, class_token, pro_mask, structure)
                batch_pred.append(output)
                all_attn_weight[i].append(cross_attn_weight)
                batch_attn_weight.append(cross_attn_weight)
            #average the predictions and attention weights across nine models within each batch
            all_preds = torch.cat([all_preds, torch.stack(batch_pred)], dim = 1) #all_preds: [9, B, 6]
            mean_batch_attn_weight = torch.mean(torch.stack(batch_attn_weight), dim=0)
            mean_attn_weight.append(mean_batch_attn_weight)
            #attention_analysis
            true_np = biosyn_class.detach().cpu().numpy()
            pred_np = np.round(torch.sigmoid(output).detach().cpu().numpy())
            selected_gene_kind, selected_pfam = attention_analysis(true_np, pred_np, gene_kind, pfam, mean_batch_attn_weight, attention_percent = 0.2) #select gene kind and pfam with high attention scores
            for l1, l2, l3, l4 in zip(all_gene_kind, selected_gene_kind, all_pfam, selected_pfam):
                l1.extend(l2)
                l3.extend(l4)

            true = torch.cat((true, biosyn_class), dim = 0)
        all_preds = torch.sigmoid(all_preds)    
    if mean_result:
        mean_pred = torch.mean(all_preds, dim = 0)
        metrics = calculate_metrics(true, mean_pred)

        for i in range(6):
            gene_kind_counter = Counter(all_gene_kind[i])
            gene_kind_high_attn.append(gene_kind_counter)
            all_pfam_processed=[elem for item in all_pfam[i] for elem in (item if isinstance(item, tuple) else [item])]
            pfam_counter=Counter(all_pfam_processed)
            pfam_counter.pop("error", None)
            pfam_high_attn.append(pfam_counter)

        if ensemble_dir is not None:
            torch.save(mean_pred, os.path.join(ensemble_dir, "ensemble_pred_test.pth"))
            torch.save(metrics, os.path.join(ensemble_dir, "ensemble_metrics_test.pth"))
            torch.save(mean_attn_weight, os.path.join(ensemble_dir, "ensemble_mean_attn_test.pth"))
            torch.save(gene_kind_high_attn, os.path.join(ensemble_dir, "gene_kind_high_attn.pth"))
            torch.save(pfam_high_attn, os.path.join(ensemble_dir, "pfam_high_attn.pth"))
            torch.save(true, os.path.join(ensemble_dir, "true_test.pth"))

        return metrics, true, mean_pred, mean_attn_weight, gene_kind_high_attn, pfam_high_attn
    else:
        #all preds : [9, N_BGC, 6]
        for i in range(all_preds.size(0)):
            model_metrics = calculate_metrics(true, all_preds[i])
            for key in metrics:
                metrics[key].append(model_metrics[key])
        if ensemble_dir is not None:
            torch.save(all_preds, os.path.join(ensemble_dir, "individual_pred_test.pth"))
            torch.save(metrics, os.path.join(ensemble_dir, "individual_metrics_test.pth"))
            torch.save(all_attn_weight, os.path.join(ensemble_dir, "individual_attn_test.pth"))
        return metrics, true, all_preds, all_attn_weight

def kensemble_MAPtest(models, test_loader, ensemble_dir = None, average_cross_attn=True, mean_result = True):
    device = next(models[0].parameters()).device
    true = torch.empty((0,1), device = device)
    all_preds = torch.empty((len(models), 0, 1), device = device)
    all_attn_weight = [[] for i in range(len(models))]
    metrics = {"overall_auc":[],"overall_recall":[],"overall_precision":[],"f1":[]}
    mean_attn_weight = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc = "batch"):
            labels, biosyn_class, pro, sub, is_product, length, pro_mask, sub_mask, structure, gene_kind, pfam = move_to_device(device, *batch)
            batch_pred = []
            batch_attn_weight = []
            for i, model in enumerate(models):
                model.eval()
                output, cross_attn_weight = model(pro, sub, structure, pro_mask, sub_mask, average_cross_attn)
                #output: (batch_size,1), mean_attn_weight:(BGC_len, smiles_len)
                batch_pred.append(output)
                all_attn_weight[i].append(cross_attn_weight)
                batch_attn_weight.append(cross_attn_weight)
            all_preds = torch.cat([all_preds, torch.stack(batch_pred)], dim = 1) #all_preds: [9, B, 1]
            mean_attn_weight.append(torch.stack(batch_attn_weight).mean(dim = 0))
            is_product = is_product.unsqueeze(-1)
            true = torch.cat((true, is_product), dim = 0)
        all_preds = torch.sigmoid(all_preds)
    if mean_result:
        mean_pred = torch.mean(all_preds, dim = 0)
        recall = recall_cal(true, mean_pred, 0.5)
        precision = precision_cal(true, mean_pred, 0.5)
        metrics["overall_auc"] = auc_cal(true, mean_pred, per_class=False)
        metrics["overall_recall"] = recall
        metrics["overall_precision"] = precision
        metrics["f1"] = f1_cal(recall, precision, mode="micro")
        if ensemble_dir is not None:
            torch.save(mean_pred, os.path.join(ensemble_dir, "ensemble_pred_test.pth"))
            torch.save(metrics, os.path.join(ensemble_dir, "ensemble_metrics_test.pth"))
            torch.save(true, os.path.join(ensemble_dir, "true_test.pth"))
            torch.save(mean_attn_weight, os.path.join(ensemble_dir, "attn_weight.pth"))
        return metrics, true, mean_pred, mean_attn_weight
    else:
        for i in range(all_preds.size(0)):
            recall = recall_cal(true, all_preds[i], 0.5)
            precision = precision_cal(true, all_preds[i], 0.5)
            metrics["overall_auc"].append(auc_cal(true, all_preds[i], per_class=False))
            metrics["overall_recall"] = recall
            metrics["overall_precision"] = precision
            metrics["f1"] = f1_cal(recall, precision, mode="micro")
        if ensemble_dir is not None:
            torch.save(all_preds, os.path.join(ensemble_dir, "individual_pred_test.pth"))
            torch.save(metrics, os.path.join(ensemble_dir, "individual_metrics_test.pth"))
        return metrics, true, all_preds
    
def predict_MAC(models, predict_loader):
    device = next(models[0].parameters()).device
    with torch.no_grad():
        all_preds = torch.empty((len(models),0,6), device=device)
        all_attn_weight = [[] for i in range(len(models))]
        mean_attn_weight = []
        for batch in tqdm(predict_loader, desc="batch"): # each batch
            batch_pred = []
            batch_attn_weight = []
            label, biosyn_class, pro, length, pro_mask, structure, class_token, gene_kind, pfam = move_to_device(device,*batch)
            for i, model in enumerate(models):
                model.eval()
                output, cross_attn_weight = model(pro, class_token, pro_mask, structure)
                batch_pred.append(output)
                all_attn_weight[i].append(cross_attn_weight)
                batch_attn_weight.append(cross_attn_weight)
            #average the predictions and attention weights across nine models within each batch
            all_preds = torch.cat([all_preds, torch.stack(batch_pred)], dim = 1) #all_preds: [9, B, 6]
            mean_batch_attn_weight = torch.mean(torch.stack(batch_attn_weight), dim=0)
            mean_attn_weight.append(mean_batch_attn_weight)
    return torch.sigmoid(all_preds), mean_attn_weight, all_attn_weight

def predict_MAP(models, predict_loader):
    device = next(models[0].parameters()).device
    all_preds = torch.empty((len(models), 0, 1), device = device)
    all_attn_weight = [[] for i in range(len(models))]
    mean_attn_weight = []
    with torch.no_grad():
        for batch in tqdm(predict_loader, desc = "batch"):
            labels, biosyn_class, pro, sub, is_product, length, pro_mask, sub_mask, structure, gene_kind, pfam = move_to_device(device, *batch)
            batch_pred = []
            batch_attn_weight = []
            for i, model in enumerate(models):
                model.eval()
                output, cross_attn_weight = model(pro, sub, structure, pro_mask, sub_mask, True)
                #output: (batch_size,1), mean_attn_weight:(BGC_len, smiles_len)
                batch_pred.append(output)
                all_attn_weight[i].append(cross_attn_weight)
                batch_attn_weight.append(cross_attn_weight)
            all_preds = torch.cat([all_preds, torch.stack(batch_pred)], dim = 1) #all_preds: [9, B, 1]
            mean_attn_weight.append(torch.stack(batch_attn_weight).mean(dim = 0))
    return torch.sigmoid(all_preds), mean_attn_weight, all_attn_weight

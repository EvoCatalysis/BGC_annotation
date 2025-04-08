import torch
import sys
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data_preparation.BGCdataset import MACDataset, MAPDataset
import random

class_dataloaders = { 'other': [],  'ribosomal': [], 'saccharide': [], 'terpene': []}

def generate_kfold(k, data, config):
  if config.task == "classification":
     dataset = MACDataset
     collate_fn = MAC_collate_fn
     val_trues_column = "biosyn_class"
  elif config.task == "product_matching":
     dataset = MAPDataset
     collate_fn = MAP_collate_fn
     val_trues_column = "is_product"
  else:
        raise ValueError(f"Unknown task: {config.task}")
  data = data.sample(frac=1, random_state = config.random_seed).reset_index(drop=True)
  folds = np.array_split(data, k+1) 
  val_trues = []
  test_fold = folds[-1]
  train_val_folds = folds[:-1]
  train_val_data = pd.concat(train_val_folds, ignore_index=True)
  kf = KFold(n_splits=k, shuffle=True, random_state = config.random_seed)
  test_dataset = dataset.from_df(test_fold, config.use_structure)
  test_dataloader = DataLoader(test_dataset,batch_size = config.test_bsz,collate_fn=collate_fn)

  dataloaders = [test_dataloader]
  for train_index, val_index in kf.split(train_val_data):
    train_data = train_val_data.iloc[train_index]
    val_data = train_val_data.iloc[val_index]
    if config.task == "classification":
      val_trues.append(torch.stack(val_data[val_trues_column].to_list()))
    else: 
       val_trues.append(torch.Tensor(val_data[val_trues_column].to_list()))
    train_dataset = dataset.from_df(train_data, config.use_structure)
    val_dataset = dataset.from_df(val_data, config.use_structure)
    train_dataloader = DataLoader(train_dataset, batch_size = config.train_bsz, shuffle = True, collate_fn = collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size = config.validation_bsz, collate_fn = collate_fn)

    dataloaders.append({
        'train': train_dataloader,
        'val': val_dataloader,
    })
  return dataloaders, test_fold, val_trues

def MAC_collate_fn(batch):
    labels, biosyn_class, protein_reps, length, structure, class_token, gene_kind, pfam = zip(*batch)
    protein_reps = [torch.stack(seq) for seq in protein_reps]
    protein_reps_padded = pad_sequence(protein_reps,batch_first=True,padding_value=0)
    protein_mask = (protein_reps_padded==0).any(dim=-1) #padding mask
    biosyn_class = torch.stack(biosyn_class).float() if None not in biosyn_class else None
    class_token = torch.stack(class_token) if None not in class_token else None
    structure_padded = None
    if None not in structure:
      structure = [torch.stack(struct) for struct in structure]
      structure_padded = pad_sequence(structure,batch_first=True,padding_value=0)
    return labels, biosyn_class, protein_reps_padded, length, protein_mask, structure_padded, class_token, gene_kind,pfam

def MAP_collate_fn(batch):
    vocab_size = 138
    labels, biosyn_class, protein_reps, subs, is_products, length, structure, gene_kind, pfam = zip(*batch)
    protein_reps = [torch.stack(seq) for seq in protein_reps]
    protein_reps_padded=pad_sequence(protein_reps,batch_first=True,padding_value=0)
    protein_mask=(protein_reps_padded==0).any(dim=-1) #padding mask
    subs_padded=pad_sequence(subs,batch_first=True, padding_value = vocab_size)
    sub_mask=(subs_padded == vocab_size)
    structure_padded = None
    is_products = torch.tensor(is_products).float()
    if None not in structure:
      structure = [torch.stack(struct) for struct in structure]
      structure_padded = pad_sequence(structure, batch_first=True, padding_value=0)
    return labels, biosyn_class, protein_reps_padded, subs_padded, is_products, length, protein_mask, sub_mask, structure_padded, gene_kind, pfam

def generate_leave_out(BGC_data, biosyn_class, train_frac, config):
  collate_fn = MAP_collate_fn
  test_data=BGC_data[BGC_data["biosyn_class"].apply(lambda x:biosyn_class in x)] #test
  leave_out_data=BGC_data[BGC_data["biosyn_class"].apply(lambda x:biosyn_class not in x)]
  leave_out_data=leave_out_data.sample(frac=1,random_state=42).reset_index(drop=True)
  train_size = int(train_frac * len(leave_out_data))
  val_size = len(leave_out_data)-train_size
  train_data = leave_out_data[:train_size]
  val_data = leave_out_data[train_size:]

  test_dataset = MAPDataset.from_df(test_data, config.use_structure, config.use_unimol)
  test_dataloader=DataLoader(test_dataset, batch_size = config.test_bsz, collate_fn=collate_fn)
  train_dataset = MAPDataset.from_df(train_data, config.use_structure, config.use_unimol)
  val_dataset = MAPDataset.from_df(val_data, config.use_structure, config.use_unimol)
  train_dataloader = DataLoader(train_dataset, batch_size = config.train_bsz, shuffle = True, collate_fn = collate_fn)
  val_dataloader = DataLoader(val_dataset, batch_size = config.validation_bsz, collate_fn = collate_fn)
  dataloaders=[test_dataloader,{
        'train': train_dataloader,
        'val': val_dataloader,
    }]
  return dataloaders
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data_preparation.BGCdataset import MACDataset, MAPDataset
import random
from functools import partial

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
  test_dataloader = DataLoader(test_dataset,batch_size = config.test_bsz,collate_fn=partial(collate_fn, is_training = False))

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
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size = config.train_bsz, 
                                  shuffle = True, 
                                  collate_fn = partial(collate_fn, is_training = True))
    val_dataloader = DataLoader(val_dataset, 
                                batch_size = config.validation_bsz, 
                                collate_fn = partial(collate_fn, is_training = False))

    dataloaders.append({
        'train': train_dataloader,
        'val': val_dataloader,
    })
  return dataloaders, test_fold, val_trues

def MAC_collate_fn(batch, is_training = None):
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
    return {
            "labels": labels,
            "biosyn_class": biosyn_class,
            "protein_reps_padded": protein_reps_padded,
            "length": length,
            "protein_mask": protein_mask,
            "structure_padded": structure_padded,
            "class_token": class_token,
            "gene_kind": gene_kind,
            "pfam": pfam
            }

def MAP_collate_fn(batch, is_training):
    vocab_size = 138

    if is_training:
      # Randomly sample new negative pairs
      new_batch = []
      pos_indices = [i for i,item in enumerate(batch) if item[4] == 1]
      neg_indices = [i for i,item in enumerate(batch) if item[4] == 0]
      used_neg_indices = set()
      for pos_idx in pos_indices:
        available_negs = [i for i in neg_indices if i not in used_neg_indices]
        if not available_negs:
          break
        
        selected_neg_idx = random.choice(available_negs)

        pos_item = batch[pos_idx]
        neg_item = batch[selected_neg_idx]
        
        if torch.equal(pos_item[3], neg_item[3]):
          continue
        used_neg_indices.add(selected_neg_idx)
        new_item = pos_item[:3] + neg_item[3:5]+ pos_item[5:]
        new_batch.append(tuple(new_item))

      remained_indices = [i for i in range(len(batch)) 
                        if i not in used_neg_indices]
      final_batch = [batch[i] for i in remained_indices] + new_batch
    
    final_batch = batch
    labels, biosyn_class, protein_reps, sub, is_product, length, structure, gene_kind, pfam = zip(*final_batch)
    # seq: 1280dim tensor
    # protein_reps: [B, N_BGC, 1280]
    protein_reps = [torch.stack(seq) for seq in protein_reps]
    protein_reps_padded=pad_sequence(protein_reps, batch_first=True, padding_value=0)
    protein_mask=(protein_reps_padded==0).any(dim=-1) #padding mask
    sub_padded=pad_sequence(sub, batch_first=True, padding_value = vocab_size)
    sub_mask=(sub_padded == vocab_size)
    structure_padded = None
    is_product = torch.tensor(is_product).float()
    if None not in structure:
      structure = [torch.stack(struct) for struct in structure]
      structure_padded = pad_sequence(structure, batch_first=True, padding_value=0)

    return {
      "labels": labels,    # BGC_number                
      "biosyn_class": biosyn_class,        
      "protein_reps_padded": protein_reps_padded,  
      "sub_padded": sub_padded,         
      "is_product": is_product,          
      "length": length,                    
      "protein_mask": protein_mask,       
      "sub_mask": sub_mask,                
      "structure_padded": structure_padded,   
      "gene_kind": gene_kind,              
      "pfam": pfam                         
      }

def generate_leave_out(BGC_data, biosyn_class, train_frac, config):
  collate_fn = MAP_collate_fn
  test_data=BGC_data[BGC_data["biosyn_class"].apply(lambda x:biosyn_class in x)] #test
  leave_out_data=BGC_data[BGC_data["biosyn_class"].apply(lambda x:biosyn_class not in x)]
  leave_out_data=leave_out_data.sample(frac=1,random_state=42).reset_index(drop=True)
  train_size = int(train_frac * len(leave_out_data))
  train_data = leave_out_data[:train_size]
  val_data = leave_out_data[train_size:]

  test_dataset = MAPDataset.from_df(test_data, config.use_structure)
  test_dataloader=DataLoader(test_dataset, batch_size = config.test_bsz, collate_fn=partial(collate_fn, is_training = False))
  train_dataset = MAPDataset.from_df(train_data, config.use_structure)
  val_dataset = MAPDataset.from_df(val_data, config.use_structure)
  train_dataloader = DataLoader(train_dataset, batch_size = config.train_bsz, shuffle = True, collate_fn = partial(collate_fn, is_training = False))
  val_dataloader = DataLoader(val_dataset, batch_size = config.validation_bsz, collate_fn = partial(collate_fn, is_training = False))
  dataloaders=[test_dataloader,{
        'train': train_dataloader,
        'val': val_dataloader,
    }]
  return dataloaders
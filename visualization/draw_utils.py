import os
import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from sklearn.metrics import roc_curve, auc, roc_auc_score

class_dict = { 'NRPS': 0, 'other': 1, 'PKS': 2, 'ribosomal': 3, 'saccharide': 4, 'terpene': 5}
class_dict_2 = { 'NRP': 0, 'Other': 1, 'Polyketide': 2, 'RiPP': 3, 'Saccharide': 4, 'Terpene': 5}


def count_gene_kind(BGC_data,verbose=0)->list:
  """
  Analyzes the distribution of gene kinds across different biosynthetic gene cluster (BGC) classes.
  Args:
      BGC_data (pd.DataFrame):
          A pandas DataFrame containing information about biosynthetic gene clusters, where each row represents a BGC.
          The DataFrame is expected to have the following columns:
              - biosyn_class: A list or array indicating the presence of the BGC in each of seven classes (binary values).
              - gene_kind: A list of gene kinds associated with the BGC.
      verbose (int, optional):
          If set to 1, prints detailed statistics about the distribution of gene kinds for each class. Default is 0.

  Returns:
      list: A list containing three elements:
          - result_counter (list[dict]): A list of dictionaries, where each dictionary contains the count of each gene kind for a specific BGC class.
          - result_percents (list[dict]): A list of dictionaries, where each dictionary contains the percentage distribution of gene kinds for a specific BGC class.
          - total_num (list[int]): A list of integers representing the total count of gene kinds for each BGC class.
  """
  result=[[] for _ in range(6)]
  for row in BGC_data.itertuples(index=True):
    indices = (row.biosyn_class == 1).nonzero(as_tuple=True)[0]
    for idx in indices:
        result[idx].extend(row.gene_kind)
  result_counter=[]
  result_percents=[]
  total_num=[]
  for i in range(6):
      gene_kind_counter = Counter(result[i])
      result_counter.append(dict(gene_kind_counter))
      total = sum(gene_kind_counter.values())
      total_num.append(total)
      percentages = {key: (value / total) * 100 for key, value in gene_kind_counter.items()}
      result_percents.append(percentages)
      if verbose:
        print(['NRPS', 'other', 'PKS', 'ribosomal', 'saccharide', 'terpene'][i],":")
        print(percentages)
        print(gene_kind_counter)
  return result_counter,result_percents,total_num

def count_pfam(BGC_data,verbose=0)->list:
  """
  Analyzes the distribution of PFAM domains across different biosynthetic gene cluster (BGC) classes.
  Args:
      BGC_data (pd.DataFrame):
          A pandas DataFrame containing information about biosynthetic gene clusters, where each row represents a BGC.
          The DataFrame is expected to have the following columns:
              - biosyn_class: A list or array indicating the presence of the BGC in each of seven classes (binary values).
              - pfam: A list of PFAM domains (or tuples of domains) associated with the BGC.
      verbose (int, optional):
          If set to 1, prints detailed statistics about the distribution of PFAM domains for each class. Default is 0.

  Returns:
      list: A list containing three elements:
          - result_counter (list[dict]): A list of dictionaries, where each dictionary contains the count of each PFAM domain for a specific BGC class.
          - result_percents (list[dict]): A list of dictionaries, where each dictionary contains the percentage distribution of PFAM domains for a specific BGC class.
          - total_num (list[int]): A list of integers representing the total count of PFAM domains for each BGC class.
  """
  result=[[] for _ in range(6)]
  for row in BGC_data.itertuples(index=True):
    indices = (row.biosyn_class == 1).nonzero(as_tuple=True)[0] #选中对应类别的index编号。
    for idx in indices:
        pfam=[item for element in row.pfam for item in (element if isinstance(element, tuple) else (element,))]
        result[idx].extend(pfam)
  result_counter=[]
  result_percents=[]
  total_num=[]
  for i in range(6):
      pfam_counter = Counter(result[i])
      pfam_counter.pop("error",None)
      result_counter.append(dict(pfam_counter))
      total = sum(pfam_counter.values())
      total_num.append(total)
      percentages = {key: (value / total) * 100 for key, value in pfam_counter.items()}
      result_percents.append(percentages)
      if verbose:
        print(['NRPS', 'other', 'PKS', 'ribosomal', 'saccharide', 'terpene'][i],":")
        print(percentages)
        print(pfam_counter)
  return result_counter, result_percents, total_num


def tensor_to_columns(df:pd.DataFrame, column_name:str):
    col_names = ['NRP', 'Other', 'Polyketide', 'RiPPs', 'Saccharide', 'Terpene']

    tensor_values = []
    for tensor in df[column_name]:
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor_values.append(tensor.detach().numpy())
        else:
            tensor_values.append(np.array(tensor))

    expanded_df = pd.DataFrame(
        np.stack(tensor_values, axis=0), 
        columns=col_names, 
        index=df.index
    )
   
    result = pd.concat([df.drop(column_name, axis=1), expanded_df], axis=1)


    return result

def tensor_to_classes(input_data, threshold=0.5):
    """

    input_data : torch.Tensor, np.ndarray, list
    threshold : float, default=0.5
    Return: stre

    Example:
    >>> tensor_to_classes([0.9, 0.1, 0, 0, 0, 0.9])
    'NRP-Terpene'
    >>> tensor_to_classes(torch.tensor([0.6, 0.7, 0.3, 0.8, 0.2, 0.4]))
    'NRP-Other-RiPPs'
    """
    class_mapping = {
        0: 'NRP',
        1: 'Other', 
        2: 'Polyketide', 
        3: 'RiPP',
        4: 'Saccharide',
        5: 'Terpene'
    }
    
    if isinstance(input_data, torch.Tensor):
        if input_data.is_cuda:
            input_data = input_data.cpu()
        values = input_data.detach().numpy()
    elif isinstance(input_data, (list, tuple)):
        values = np.array(input_data)
    elif isinstance(input_data, np.ndarray):
        values = input_data
    else:
        raise ValueError(f"False type: {type(input_data)}")
    
    if values.shape != (6,):
        raise ValueError(f"Expect shape(6,), but got{values.shape}")
    
    active_indices = np.where(values > threshold)[0]
    
    active_classes = [class_mapping[i] for i in active_indices]
    
    if active_classes:
        return '-'.join(active_classes)
    else:
        return ''  

def class_to_tensor(class_list,class_dict):
    class_label=[0]*len(class_dict)
    if "error" in class_list:
        return torch.tensor(class_label)
    else:
        for biosyn_class in class_list:
            index=class_dict[biosyn_class]
            class_label[index]=1
        return torch.tensor(class_label)


def get_antiSMASH_class(gbk_file):
    predicted_classes=[]
    BGC_number=os.path.splitext(os.path.basename(gbk_file))[0]
    records = SeqIO.parse(gbk_file, "genbank")
    for record in records:
        for feature in record.features:
            if feature.type=="protocluster":
                predicted_classes.extend(feature.qualifiers.get("category"))
    if predicted_classes == []:
        predicted_classes = ["error"]
    return BGC_number, list(set(predicted_classes))

def calculate_ECFP(smiles): #convert smiles to ECFP (string)
    try:
        molecule=Chem.MolFromSmiles(smiles)

        rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
        np_bits = rdkgen.GetFingerprintAsNumPy(molecule)

    except:
        return None
    return np_bits

def get_position(gbk_file):
    record = SeqIO.read(gbk_file, "genbank")
    start = record.annotations["structured_comment"]["antiSMASH-Data"]["Orig. start"]
    end = record.annotations["structured_comment"]["antiSMASH-Data"]["Orig. end"]
    return int(start), int(end)

def find_gbk(row, base_dir):
    start = int(row['From'])
    end = int(row['To'])
    accession = row['NCBI accession']
    file_list = os.listdir(base_dir)
    target_file_list = [f for f in file_list if f.startswith(accession) and f.endswith('.gbk')]
    for file_name in target_file_list:
        gbk_file = os.path.join(base_dir, file_name)
        if get_position(gbk_file) == (start, end):
            return gbk_file
    return None

def cal_metrics(pred:np.array, true:np.array, mask = None):
    metrics = {
    'precision': [],
    'recall': [],
    'f1': [],
    "AUC": [],
    }
    
    pred_round = np.round(pred)
    if mask is None:
        mask = np.ones(true.shape[0])
    for i in range(pred.shape[1]): 
        tp = np.sum(((pred_round[:, i] == 1) & (true[:, i] == 1)) * mask)  
        fn = np.sum(((pred_round[:, i] == 0) & (true[:, i] == 1)) * mask)  
        fp = np.sum(((pred_round[:, i] == 1) & (true[:, i] == 0)) * mask)  
        tn = np.sum(((pred_round[:, i] == 0) & (true[:, i] == 0)) * mask)  

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0 
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if mask is not None:
            roc_auc = roc_auc_score(true[mask.astype(bool)][:, i], pred[mask.astype(bool)][:, i])
        else:
            roc_auc = roc_auc_score(true[:, i], pred[:, i])

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics["f1"].append(f1)
        metrics["AUC"].append(roc_auc)
    return metrics


def extract_deepbgc_tsv(tsv_file_path):

    df = pd.read_csv(tsv_file_path, sep='\t')
    
    first_row = df.iloc[0]
    
    alkaloid = first_row.get('Alkaloid', 0)
    nrp = first_row.get('NRP', 0)
    other = first_row.get('Other', 0)
    polyketide = first_row.get('Polyketide', 0)
    ripp = first_row.get('RiPP', 0)
    saccharide = first_row.get('Saccharide', 0)
    terpene = first_row.get('Terpene', 0)
    
    alkaloid_max = max(alkaloid, other)
    
    result = torch.tensor([nrp, alkaloid_max, polyketide, ripp, saccharide, terpene])
    
    return result


def compute_confusion_matrix(true, pred):

    row_sums = true.sum(axis = 1)
    valid_rows = row_sums == 1  

    true_filtered = true[valid_rows]
    pred_filtered = pred[valid_rows]

    num_classes = true.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes + 1), dtype=int)

    for true_row, pred_row in zip(true_filtered, pred_filtered):
        true_class = np.argmax(true_row).item() 

        pred_classes = np.where(pred_row == 1)[0].tolist()

        if not pred_classes:  
            confusion_matrix[true_class, -1] += 1
        else:  
            for pred_class in pred_classes:
                confusion_matrix[true_class, pred_class] += 1

    return confusion_matrix
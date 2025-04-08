import os
import numpy as np
import glob
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import torch
import pickle
from tqdm import tqdm
from collections import Counter


def generate_negatives(lower:np.ndarray,
                       higher:float,
                       i:int,
                       similarity_matrix:np.ndarray,
                       smiles_group:dict,
                       smiles_list:list,
                       coconut_fps:list,
                       data_cleaned:pd.DataFrame): 
    """
    Generates negative examples by randomly sampling natural products from the coconut database 
    Args:
        lower (numpy.ndarray): A numpy array of lower similarity thresholds (calculated per row).
                               Each element corresponds to a certain percentile of the similarity scores for a product. 
                               即相似度矩阵 similarity_matrix 中每一行的第 x 百分位数的值(x由用户指定)。如第 99.9 百分位数表示比 99.9% 的数据更大或相等的值。
        higher (float): An upper similarity threshold. Products with similarity scores below this value are eligible.
        i (int): The number of negative examples to generate for each BGC number-positive product pair.
        similarity_matrix (numpy.ndarray): A 2D similarity matrix (shape: [n, n]) where each row represents
                                            similarity scores between a product and all other products.
        smiles_group (dict): A dictionary mapping SMILES strings to their associated BGC numbers.
                             Key: SMILES string, Value: List of BGC numbers.
                             {smiles:[BGC1,BGC2]}
        smiles_list (list): A list of SMILES strings corresponding to rows in the similarity matrix.
    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a negative example. Each dictionary includes:
    Example:
        result = generate_negatives(lower, higher, 3, similarity_matrix, smiles_group, smiles_list)
        # Generates negative examples with 3 samples for each BGC-product pair.
    Logic:
        1. Iterate through each product in the similarity matrix.
        2. For each product:
           a. Identify indices of products with similarity scores within the range [lower[product], higher].
           b. Retrieve the associated BGC numbers for the current product.
           c. Determine the number of negative examples to generate (n = i * number of BGCs).
           d. Randomly sample `n` indices from the eligible products.
           e. For each sampled index:
              - Create a dictionary containing information about the BGC number, selected product, biosynthetic class,
                enzyme list, negative example flag (`is_product=0`)
        3. Append each dictionary to the result list.
        4. Return the complete list of negative examples.
    Notes:
        - Each BGC number-positive product pair is assigned `i` negative samples.
        - Random sampling (`np.random.choice`) is done without replacement (`replace=False`).
        - The function assumes the input data structures (e.g., `coconut_fps` and `data_cleaned`) are globally accessible.
    """
    result=[]
    length=similarity_matrix.shape[0] 
    for product in tqdm(range(length), desc="mibig_smiles"):
        row = similarity_matrix[product]
        indices = np.where((row >= lower[product]) & (row <= higher))[0]
        BGCs = smiles_group[smiles_list[product]] 
        n = i*len(BGCs)
        selected_indices = np.random.choice(indices, n, replace=False)
        for j in range(n):
            result_dict={}
            BGC_num=BGCs[j//i] 
            #Each BGC_number-positive_product pair is assigned i negative samples
            result_dict["BGC_number"]=BGC_num
            result_dict["product"]=coconut_fps[selected_indices[j]][1]
            result_dict["biosyn_class"]=data_cleaned.loc[data_cleaned["BGC_number"]==BGC_num,"biosyn_class"].values[0]
            result_dict["enzyme_list"]=data_cleaned.loc[data_cleaned["BGC_number"]==BGC_num,"enzyme_list"].values[0]
            result_dict["is_product"]=0
            result.append(result_dict)
    return result

def count_gene_kind(BGC_data:pd.DataFrame, verbose = 0)->list:
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

def count_pfam(BGC_data:pd.DataFrame, verbose=0)->list:
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
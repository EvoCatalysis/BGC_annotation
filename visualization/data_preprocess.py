import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import torch
import shutil
from tqdm import tqdm
target_folder = "e:\\project\\np\\Natural_product"
os.chdir(target_folder)
CURRENT_DIR=os.getcwd()

class_dict = { 'NRPS': 0, 'other': 1, 'PKS': 2, 'ribosomal': 3, 'saccharide': 4, 'terpene': 5}
data_cleaned = pd.read_pickle(os.path.join(CURRENT_DIR,"data","BGC_4.0","BGC_structure4.pkl"))
data_unique = pd.read_pickle(os.path.join(CURRENT_DIR,"data","BGC_4.0","BGC_data_unique4.pkl"))
data_nostructure = pd.read_pickle(os.path.join(CURRENT_DIR,"data","BGC_4.0","mibig_no_structure4.pkl"))
similarity_matrix = pd.read_pickle(os.path.join(CURRENT_DIR,"data","natural_product","similarity_matrix4.pkl"))
coconut_fps = pd.read_pickle(os.path.join(CURRENT_DIR,"data","natural_product","coconut_mol.pkl"))
smiles_group = data_cleaned.groupby("product")["BGC_number"].apply(list).to_dict()
smiles_dict = data_cleaned.set_index("product")["BGC_number"].to_dict()
mibig_smiles_list = list(smiles_dict.keys())

def class_to_index(class_list,class_dict):
    class_label=[0]*len(class_dict)
    if "error" in class_list:
        return torch.tensor(class_label)
    else:
        for biosyn_class in class_list:
            index=class_dict[biosyn_class]
            class_label[index]=1
        return torch.tensor(class_label)

def generate_negatives(lower,higher,i,similarity_matrix,smiles_group,smiles_list): 
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
    length=similarity_matrix.shape[0] #3551
    for product in tqdm(range(length), desc="mibig_smiles"):
        row = similarity_matrix[product]
        indices = np.where((row >= lower[product]) & (row <= higher))[0]
        BGCs = smiles_group[smiles_list[product]] #找到该product对应的所有BGC
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

def get_rep_dict(representations): #convert enzyme representations (esm1b) to dictionary. Representations is a list with tuple as elements
    #deprecated
    rep_dict={}
    for _,key,value in representations: #[(layer,BGC_number,representation)]
        if key not in rep_dict: #rep_dict: {BGC_number:[rep1,rep2,rep3...]}
            rep_dict[key]=[]
        rep_dict[key].append(value)
    return rep_dict

def generate_train_val(data,train_frac,train_path,val_path): #divide the data into train set and validation set
    import random
    data=data.copy()
    data=data.sample(frac=1,random_state=random.randint(1, 100000)).reset_index(drop=True)
    train_size = int(train_frac * len(data))
    val_size = len(data) - train_size 

    train_data = data[:train_size]
    val_data = data[train_size:]

    train_data.to_pickle(train_path)
    val_data.to_pickle(val_path)
    print("successfully saved",f"train_size:{train_size},val_size:{val_size}")

def calculate_ECFP(smiles): #convert smiles to ECFP (string)
    try:
        molecule=Chem.MolFromSmiles(smiles)
        ECFP=AllChem.GetMorganFingerprintAsBitVect(molecule, 3, nBits=1024)
        fp_array = np.zeros((1,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(ECFP, fp_array)
        fp_string = ''.join(map(str, fp_array))
    except:
        fp_string="error"
        print(smiles,"error")
    return fp_string

def extract_BGCgbk(pkl,gbk_path,output_path):
    data=pd.read_pickle(pkl)
    BGC_list=data["BGC_number"].to_list()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for BGC in BGC_list:
        source_file=os.path.join(gbk_path,BGC+".gbk")
        if os.path.exists(source_file):
            shutil.copy(source_file, output_path)
            print(f"copy: {BGC}.gbk to {output_path}")
        else:
            print(f"file not found: {BGC}.gbk")


if __name__=="main":
    print("Ture")
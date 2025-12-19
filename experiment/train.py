import pickle
import pandas as pd
import os
import torch
from experiment.dataloaders import generate_kfold
from experiment.ensemble_utils import kensemble_validation
import hydra
from pathlib import Path
import re
import argparse
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent

def pfam_preprocess(pfam_list) -> list: 
    #[[(pfam_num,pfam_name,E-value),()],[]]->[(pfam_name1,pfam_name2),(),]
    def process_element(element):
        if isinstance(element, list): #[(),(),]->()
            return tuple(triple[1] for triple in element)
        elif isinstance(element, tuple): #(error,error,error)->error
            return element[1]
        else:
            return element
    return [process_element(item) for item in pfam_list] #item:list

def class_to_index(class_list, class_dict):
    class_label=[0]*len(class_dict)
    for biosyn_class in class_list:
        index=class_dict[biosyn_class]
        class_label[index]=1
    return torch.tensor(class_label)

def get_smiles_index(smiles):
    if not hasattr(get_smiles_index, "dict"):
        get_smiles_index.dict = {'#': 0, '%10': 1, '%11': 2, '%12': 3, '%13': 4, '%14': 5, '%15': 6, '%16': 7, '%17': 8, '%18': 9, '%19': 10, '%20': 11, '%21': 12, '%22': 13, '%23': 14, '%24': 15, '%25': 16, '%26': 17, '(': 18, ')': 19, '.': 20, '1': 21, '2': 22, '3': 23, '4': 24, '5': 25, '6': 26, '7': 27, '8': 28, '9': 29, '=': 30, 'B': 31, 'Br': 32, 'C': 33, 'Cl': 34, 'F': 35, 'I': 36, 'N': 37, 'O': 38, 'P': 39, 'S': 40, '[55Fe+3]': 41, '[As+]': 42, '[As]': 43, '[B-]': 44, '[B@-]': 45, '[B@@-]': 46, '[BH4+3]': 47, '[BrH+]': 48, '[C+]': 49, '[C-]': 50, '[C@@H2]': 51, '[C@@H]': 52, '[C@@]': 53, '[Fe+]': 65, '[Fe-3]': 66, '[Fe-4]': 67, '[Fe@@]': 68, '[FeH2-2]': 69, '[Fe]': 70, '[H+]': 71, '[H]': 72, '[I+]': 73, '[IH2]': 74, '[K]': 75, '[Mo+2]': 76, '[N+3]': 77, '[N+]': 78, '[N-]': 79, '[N@+]': 80, '[N@@+]': 81, '[N@@H+]': 82, '[N@@]': 83, '[N@H+]': 84, '[N@]': 85, '[NH+]': 86, '[NH-]': 87, '[NH2+2]': 88, '[NH2+]': 89, '[NH2]': 90, '[NH3+]': 91, '[NH3]': 92, '[NH]': 93, '[N]': 94, '[Na+]': 95, '[Na]': 96, '[Ni]': 97, '[O+]': 98, '[O-]': 99, '[O@@]': 100, '[OH+]': 101, '[OH2+]': 102, '[O]': 103, '[P+]': 104, '[P@+]': 105, '[P@@H]': 106, '[P@@]': 107, '[P@H]': 108, '[P@]': 109, '[PH]': 110, '[P]': 111, '[S+]': 112, '[S-]': 113, '[S@+]': 114, '[S@@+]': 115, '[S@@]': 116, '[S@]': 117, '[SH+]': 118, '[SH]': 119, '[S]': 120, '[Se+]': 121, '[Se@@]': 122, '[SeH]': 123, '[Se]': 124, '[SiH2]': 125, '[SiH]': 126, '[Si]': 127, '[c+]': 128, '[c-]': 129, '[c@]': 130, '[cH-]': 131, '[n+]': 132, '[nH]': 133, 'c': 134, 'n': 135, 'o': 136, 's': 137}
    SMI_REGEX_PATTERN = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|\+|\\\/|:|@|\?|\>|\*|\$|\%[0-9]{2}|[0-9])"
    split = re.findall(SMI_REGEX_PATTERN, smiles)
    tokens = [get_smiles_index.dict.get(tk, len(get_smiles_index.dict)) for tk in split]
    return torch.tensor(tokens)

def df_preprocess(cfg, task, *dfs):

    mibig_rep_dict = torch.load(os.path.join(PROJECT_DIR, cfg.ESM2_reps))
    with open(os.path.join(PROJECT_DIR, cfg.pfam_data), 'rb') as f:
        BGC_domain_pfam = pickle.load(f)
    with open(os.path.join(PROJECT_DIR, cfg.gene_kind_data), 'rb') as f:
        BGC_gene_kind = pickle.load(f)
    
    if task == 'classification':
        class_dict = {'NRPS': 0, 'other': 1, 'PKS': 2, 'ribosomal': 3, 'saccharide': 4, 'terpene': 5}
        
        def process_data(data, mibig_rep_dict, pfam_dict, gene_kind_dict):
            data["protein_rep"] = data["BGC_number"].map(mibig_rep_dict)
            data["pfam"] = data["BGC_number"].map(pfam_dict)
            data["gene_kind"] = data["BGC_number"].map(gene_kind_dict)
            data["pfam"] = data["pfam"].apply(pfam_preprocess)
            data["biosyn_class"] = data["biosyn_class"].apply(class_to_index, class_dict=class_dict)
            return data
    
    elif task == 'product_matching':
        def process_data(data, mibig_rep_dict, pfam_dict, gene_kind_dict):
            data["product_index"] = data["product"].apply(get_smiles_index)
            data["protein_rep"] = data["BGC_number"].map(mibig_rep_dict)
            data["pfam"] = data["BGC_number"].map(pfam_dict)
            data["pfam"] = data["pfam"].apply(pfam_preprocess)
            data["gene_kind"] = data["BGC_number"].map(gene_kind_dict)
            return data
    
    else:
        raise ValueError(f"Unknown task_type: {task}. Expected 'classification' or 'product_matching'.")
    
    processed_dfs = []
    for df in dfs:
        processed_df = process_data(df, mibig_rep_dict, BGC_domain_pfam, BGC_gene_kind)
        processed_dfs.append(processed_df)
    
    return processed_dfs[0] if len(processed_dfs) == 1 else tuple(processed_dfs)


if __name__ == "__main__":
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MAC", help="Model type")
    args, _ = parser.parse_known_args()  

    parser.add_argument(
        "--ckpt", 
        default=f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Checkpoint directory name"
    )
    args = parser.parse_args() 

    #Please change to a relative path (according to your current working directory)
    with hydra.initialize(config_path =  os.path.join("..", "configs"), 
                          version_base="1.2"): 
        cfg = hydra.compose(config_name = "dataset", overrides=[f"BGC_{args.model}.device={device}"] )
    
    if args.model == "MAC":
        model_cfg = cfg.BGC_MAC
        BGC_data = pd.read_pickle(os.path.join(PROJECT_DIR, cfg.MAC_metadata))
        test_name = os.path.basename(cfg.MAC_metadata).split(".")[0]

    elif args.model == "MAP":
        model_cfg = cfg.BGC_MAP
        BGC_data = pd.read_pickle(os.path.join(PROJECT_DIR, cfg.MAP_metadata))
        test_name = os.path.basename(cfg.MAP_metadata).split(".")[0]
    print(model_cfg.device)
    #process dataframe
    print("begin processing dataframe")
    BGC_data = df_preprocess(cfg, model_cfg.task, BGC_data)
    print(f"dataset length: {len(BGC_data)}")
    #load configuration
    print("begin generating kfold")
    dataloaders, test_data, val_trues = generate_kfold(9, BGC_data, model_cfg.data)
    test_data.to_pickle(os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, f"test_{test_name}_{model_cfg.data.random_seed}.pkl"))
    ckpt = kensemble_validation(dataloaders, model_cfg, save_checkpoint = True, checkpoint_name = args.ckpt)
    print(ckpt["best_val_metric"])
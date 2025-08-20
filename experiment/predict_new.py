import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, List

from tqdm import tqdm
import os
import pandas as pd
import argparse
import hydra
import numpy as np
import pickle
from datetime import datetime

from data_preparation.BGCdataset import MACDataset, MAPDataset
from experiment.train import get_smiles_index
from experiment.dataloaders import MAC_collate_fn, MAP_collate_fn
from data_preparation.BGC import Bgc
from experiment.ensemble_utils import generate_ensemblelist, predict_MAC, predict_MAP
from data_preparation.esm2_emb_cal import generate_embedding

from rdkit import Chem
from functools import partial

PROJECT_DIR = Path(__file__).resolve().parent.parent
biosyn_class =['NRP', 'Other', 'Polyketdie', 'RiPP', 'Saccharide', 'Terpene']

def list_files(directory, ext = None): 
    file_paths = []  
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            if ext is not None:
                if ext == file_path.split(".")[-1]:
                    file_paths.append(file_path)
            else:
                file_paths.append(file_path)
    return file_paths


def extract_bgc(gbk_file: List[str], smiles: Union[None, List[List[str]]] = None) -> pd.DataFrame:
    """Extract BGC-MAC dataset from MIBiG JSON and GBK files."""
    if smiles is None:
        smiles = [None for x in gbk_file]

    assert len(gbk_file) == len(smiles), f"Mismatch between number of GBK files:{len(gbk_file)} and SMILES strings:{len(smiles)}"
    # Process BGC data
    columns = ["BGC_number", "product", "biosyn_class", "enzyme_list", "is_product"]
    BGC_data = pd.DataFrame(columns=columns)
    for gbk_file, smiles in tqdm(zip(gbk_file, smiles), desc="Processing BGC files"):
        mibig_BGC = Bgc(gbk_file, database = "new", product = smiles)  
        bgc_info = mibig_BGC.get_info()
        for info in bgc_info:
            BGC_data.loc[len(BGC_data)] = info

    return BGC_data

def canonize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol)
    return canonical_smiles

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Process BGC and natural product data")
    parser.add_argument("--model", default="MAC", help="Model type")
    parser.add_argument("--ckpt", default="default", help="checkpoint dir name")
    parser.add_argument("--gbk", help="gbk file (dir or file name)")
    parser.add_argument("--smiles", default=None, help="smiles string or pickle file")
    parser.add_argument("--output", default="../output", help="output dir")
    parser.add_argument("--esm_cache", default=None, help= "cache path for esm embeddings")
    args = parser.parse_args()

    smiles = None
    if args.smiles is not None:
        args.model = "MAP"

        if os.path.isfile(args.smiles):
            smiles = pickle.load(open(args.smiles, "rb"))
            smiles = [[canonize_smiles(smi) for smi in smiles_list] for smiles_list in smiles]
        else:
            smiles = [[canonize_smiles(args.smiles)]]

    with hydra.initialize(config_path=os.path.join("..", "configs"),
                          version_base="1.2"):
        cfg = hydra.compose(config_name="dataset", overrides=[f"BGC_{args.model}.device={device}"])


    cache_dir = cfg.inference.cache_dir
    cache_dir = f"{cache_dir}_{args.model}"
    if os.path.isfile(args.gbk):
        gbk_file = [args.gbk]
    else:
        gbk_file = list_files(args.gbk, ext = "gbk")
        gbk_file.extend(list_files(args.gbk, ext = "gb"))

    if len(smiles) == 1 and len(gbk_file) > 1:
        smiles = smiles * len(gbk_file)
    BGC_data = extract_bgc(gbk_file, smiles = smiles)
    BGC_number_deduplicate = (
        BGC_data.groupby("BGC_number")
        .agg({
            "biosyn_class": "first",  
            "enzyme_list": "first",  
        })
        .reset_index()
    )

    gbk_basename = os.path.basename(args.gbk)
    if args.esm_cache:
        esm_rep = pickle.load(open(args.esm_cache, 'rb'))
        print(f"load cache from {args.esm_cache}")
    else:
        esm_rep = generate_embedding(BGC_number_deduplicate, os.path.join(PROJECT_DIR, "data", "esm2_t33_650M_UR50D.pt"))
        pickle.dump(esm_rep, open(f'../data/esm_cache/{gbk_basename.split(".")[0]}_cache.pkl', 'wb'))
    
        
    if args.model == "MAC":
        BGC_data = BGC_number_deduplicate.copy()
        BGC_data["protein_rep"] = BGC_data["BGC_number"].map(esm_rep)
        model_cfg = cfg.BGC_MAC
        checkpoint_path = os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, args.ckpt)
        predict_dataset = MACDataset.from_df(BGC_data, model_cfg.data.use_structure)
        predict_loader = DataLoader(predict_dataset, batch_size=model_cfg.data.test_bsz, collate_fn=MAC_collate_fn)
        ckpt = torch.load(os.path.join(checkpoint_path, f"{args.ckpt}.ckpt"), weights_only=False)
        ensemble_model = generate_ensemblelist(ckpt)
        prediction = predict_MAC(ensemble_model, predict_loader)
        output_data = np.mean(prediction[0],axis=0) #(num_gbk, 6)

        df = pd.DataFrame(output_data, 
                 index=gbk_file, 
                 columns=biosyn_class).round(2)
        os.makedirs(args.output, exist_ok  =True)
        output_path = os.path.join(args.output, f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
        df.to_csv(output_path, index=True)
        print(f"save output to: {output_path}")

    elif args.model == "MAP":
        BGC_data["protein_rep"] = BGC_data["BGC_number"].map(esm_rep)
        assert None not in BGC_data["product"].tolist()
        BGC_data["product_index"] = BGC_data["product"].apply(get_smiles_index)
        model_cfg = cfg.BGC_MAP
        checkpoint_path = os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, args.ckpt)
        predict_dataset = MAPDataset.from_df(BGC_data, model_cfg.data.use_structure)
        predict_loader = DataLoader(predict_dataset, batch_size=model_cfg.data.test_bsz, collate_fn=partial(MAP_collate_fn, is_training = False))
        ckpt = torch.load(os.path.join(checkpoint_path, f"{args.ckpt}.ckpt"), weights_only=False)
        ensemble_model = generate_ensemblelist(ckpt)

        # all_preds, mean_attn_weight, all_attn_weight
        prediction = predict_MAP(ensemble_model, predict_loader)
        output_data = np.squeeze(np.mean(prediction[0], axis=0))
        flat_gbk = [gbk for i, gbk in enumerate(gbk_file) for _ in smiles[i]] 
        flat_smiles = [item for sublist in smiles for item in sublist]
        df = pd.DataFrame({
        'Gbk_file': flat_gbk,  
        'Prospective Product': flat_smiles,  
        'Probability': output_data 
        })
        os.makedirs(args.output, exist_ok  =True)
        output_path = os.path.join(args.output, f"{args.model}_{gbk_basename.split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
        df.to_csv(output_path, index=True)
        print(f"save output to: {os.path.abspath(output_path)}")
    
    #Example

    # python predict_new.py --gbk ../data/ranking/BGC0001790 --ckpt MAP_2025-08-15_02-04-00 --smiles "C[C@H](C(=O)N[C@]1(CN(C1=O)S(=O)(=O)O)OC)NC(=O)CC[C@H](C(=O)O)N"
    # python predict_new.py --gbk ../data/ranking/BGC0000448 --ckpt MAP_2025-08-15_02-04-00 --smiles "COC1NC2=C(C=C(OC)C(O)=C2)C(=O)N2CC(CC12)=CC"
    # python predict_new.py --gbk ../data/ranking/BGC0002209 --ckpt MAP_2025-08-15_02-04-00 --smiles "COC1=C(CO)C(O)=C(C=O)C(CCO)=C1"
    # python predict_new.py --gbk ../data/ranking/BGC0000693 --ckpt MAP_2025-08-15_02-04-00 --smiles "C1[C@@H]([C@H]([C@@H]([C@H]([C@@H]1NC(=O)[C@H](CCN)O)O)O[C@H]2[C@@H]([C@H]([C@H](O2)CO)O)O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CN)O)O)N)N"
    # python predict_new.py --gbk ../data/ranking/BGC0001007 --ckpt MAP_2025-08-15_02-04-00 --smiles "NC1=C(N=C2)C(C2=CC(C(/C=C/OC)=O)=N3)=C3C(NC(C)=O)=C1"


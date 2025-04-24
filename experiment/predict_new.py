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
from ensemble_utils import generate_ensemblelist, predict_MAC, predict_MAP
from data_preparation.esm2_emb_cal import generate_embedding

PROJECT_DIR = Path(__file__).resolve().parent.parent
biosyn_class =['NRP', 'Other', 'Polyketdie', 'RiPP', 'Saccharide', 'Terpene']

def list_files(directory, ext = None): 
    file_paths = []  
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            if ext is not None:
                if ext in file_path:
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Process BGC and natural product data")
    parser.add_argument("--model", default="MAC", help="Model type")
    parser.add_argument("--ckpt", default="default", help="checkpoint dir name")
    parser.add_argument("--gbk", help="gbk file (dir or file name)")
    parser.add_argument("--smiles", default=None, help="smiles string or pickle file")
    parser.add_argument("--output", default="../output", help="output dir")
    args = parser.parse_args()

    if args.smiles is not None:
        args.model = "MAP"
    with hydra.initialize(config_path=os.path.join("..", "configs"),
                          version_base="1.2"):
        cfg = hydra.compose(config_name="dataset", overrides=[f"BGC_{args.model}.device={device}"])

    if os.path.isfile(args.gbk):
        gbk_file = [args.gbk]
    else:
        gbk_file = list_files(args.gbk, ext = "gbk")
    if os.path.isfile(args.smiles):
        smiles = pickle.load(open(args.smiles, "rb"))
    else:
        smiles = [[args.smiles]]

    BGC_data = extract_bgc(gbk_file, smiles)
    BGC_number_deduplicate = (
        BGC_data.groupby("BGC_number")
        .agg({
            "biosyn_class": "first",  
            "enzyme_list": "first",  
        })
        .reset_index()
    )
    esm_rep = generate_embedding(BGC_number_deduplicate, os.path.join(PROJECT_DIR, "data", "esm2_t33_650M_UR50D.pt"))
    

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
        predict_loader = DataLoader(predict_dataset, batch_size=model_cfg.data.test_bsz, collate_fn=MAP_collate_fn)
        ckpt = torch.load(os.path.join(checkpoint_path, f"{args.ckpt}.ckpt"), weights_only=False)
        ensemble_model = generate_ensemblelist(ckpt)
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
        output_path = os.path.join(args.output, f"{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv")
        df.to_csv(output_path, index=True)
        print(f"save output to: {os.path.abspath(output_path)}")

import os
import torch
import hydra
import pandas as pd
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, List

import numpy as np
import numpy as np
import pickle
from tqdm import tqdm
import os
import pandas as pd
import argparse
import hydra
from omegaconf import DictConfig

from data_preparation.BGCdataset import MACDataset, MAPDataset
from experiment.train import get_smiles_index
from experiment.dataloaders import MAC_collate_fn, MAP_collate_fn
from data_preparation.BGC import Bgc
from ensemble_utils import generate_ensemblelist, predict_MAC, predict_MAP
from data_preparation.esm2_emb_cal import generate_embedding
from data_preparation.esm2_emb_cal import Esm_BGC

PROJECT_DIR = Path(__file__).resolve().parent.parent
class_dict = { 'NRP': 0, 'Other': 1, 'Polyketdie': 2, 'RiPP': 3, 'Saccharide': 4, 'Terpene': 5}

def list_files(directory): 
    file_paths = []  
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
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
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Process BGC and natural product data")
    parser.add_argument("--model", default="MAC",
                        help="Model type")
    parser.add_argument("--ckpt", default="MAC_ckpt1",
                        help="checkpoint dir name")
    parser.add_argument("--gbk",
                        help="gbk file")
    parser.add_argument("--smiles", default=None,
                        help="smiles string or pickle file")
    args = parser.parse_args()
    with hydra.initialize(config_path=os.path.join("..", "configs"),
                          version_base="1.2"):
        cfg = hydra.compose(config_name="dataset", overrides=[f"BGC_{args.model}.device={device}"])

    #gbk_file = list_files(os.path.join(PROJECT_DIR, "data","example", "new"))
    #smiles = [["CC1C[C@]23OC(=O)C4=C2OC1C(O)C3\C=C/C(=O)[C@@H](C)C[C@@H](C)C4=O", "CC1CC23OC(=O)C4=C2OC1C(O)C3\C=C/C(=O)C(C)CC(C)C4=O"],
            #["CCCC(O[C@H]1C[C@](C)(N)[C@H](O)[C@H](C)O1)C(C)C(O)C(CC)\C=C\C(O)C(C)C1C\C=C(C)\C(O)C(C)C(CC(O)C(C)C(O)CC2CC(O)C(O)C(O)(CC(O[C@@H]3O[C@H](C)[C@@H](O)[C@H](O[C@H]4C[C@@H](N)[C@H](O)[C@@H](C)O4)[C@H]3O[C@@H]3O[C@H](C)[C@@H](O)[C@H](O)[C@H]3O)C(C)CCC(O)CC(O)C\C=C(CC)\C(=O)O1)O2)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O"]]
    
    #gbk_file = gbk_file[0]
    if os.path.isfile(args.gbk):
        gbk_file = [args.gbk]
    else:
        gbk_file = list_files(args.gbk)
    if type(args.smiles) is str:
        smiles = [[args.smiles]]
    else:
        smiles = args.smiles

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
        ensemble_model = generate_ensemblelist(checkpoint_path)
        prediction = predict_MAC(ensemble_model, predict_loader)
        print(torch.mean(prediction[0],dim=0))

    elif args.model == "MAP":
        BGC_data["protein_rep"] = BGC_data["BGC_number"].map(esm_rep)
        assert None not in BGC_data["product"].tolist()
        BGC_data["product_index"] = BGC_data["product"].apply(get_smiles_index)
        model_cfg = cfg.BGC_MAP
        checkpoint_path = os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, args.ckpt)
        predict_dataset = MAPDataset.from_df(BGC_data, model_cfg.data.use_structure)
        predict_loader = DataLoader(predict_dataset, batch_size=model_cfg.data.test_bsz, collate_fn=MAP_collate_fn)
        ensemble_model = generate_ensemblelist(checkpoint_path)
        prediction = predict_MAP(ensemble_model, predict_loader)
        print(torch.mean(prediction[0], dim=0))

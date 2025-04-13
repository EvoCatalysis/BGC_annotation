import os
import torch
import hydra
import pandas as pd
from data_preparation.BGCdataset import MACDataset, MAPDataset
from experiment.dataloaders import MAC_collate_fn, MAP_collate_fn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from ensemble_utils import generate_ensemblelist, kensemble_MACtest, kensemble_MAPtest

PROJECT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Process BGC and natural product data")
    parser.add_argument("--model", default="MAC",
                        help="Model type")
    parser.add_argument("--ckpt", 
                        help="checkpoint dir name")
    parser.add_argument("--mean_result", action="store_true",
                    help="mean the result of ensemble models")
    args = parser.parse_args()
    with hydra.initialize(config_path =  os.path.join("..", "configs"), 
                          version_base="1.2"): 
        cfg = hydra.compose(config_name = "dataset", overrides=[f"BGC_{args.model}.device={device}"] )
    
    if args.model == "MAC":
        model_cfg = cfg.BGC_MAC
        checkpoint_path = os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, args.ckpt)
        test_data = pd.read_pickle(os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, f"test_{model_cfg.data.task}_{model_cfg.data.random_seed}.pkl"))
        test_dataset = MACDataset.from_df(test_data, model_cfg.data.use_structure)
        test_loader = DataLoader(test_dataset, batch_size=model_cfg.data.test_bsz, collate_fn=MAC_collate_fn)
        ckpt = torch.load(os.path.join(checkpoint_path, f"{args.ckpt}.ckpt"), weights_only=False)
        ensemble_model = generate_ensemblelist(ckpt)
        test_results = kensemble_MACtest(ensemble_model, test_loader, checkpoint_path, mean_result = args.mean_result)
        print(test_results["metrics"])
    elif args.model == "MAP":
        model_cfg = cfg.BGC_MAP
        checkpoint_path = os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, args.ckpt)
        test_data = pd.read_pickle(os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, f"test_{model_cfg.data.task}_{model_cfg.data.random_seed}.pkl"))
        test_dataset = MAPDataset.from_df(test_data, model_cfg.data.use_structure)
        test_loader = DataLoader(test_dataset, batch_size=model_cfg.data.test_bsz, collate_fn=MAP_collate_fn)
        ckpt = torch.load(os.path.join(checkpoint_path, f"{args.ckpt}.ckpt"), weights_only=False)
        ensemble_model = generate_ensemblelist(ckpt)
        test_results = kensemble_MAPtest(ensemble_model, test_loader, checkpoint_path, mean_result = args.mean_result)
        print(test_results["metrics"])


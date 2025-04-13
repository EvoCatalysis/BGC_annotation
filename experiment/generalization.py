import pickle
import pandas as pd
import os
import torch
from dataloaders import generate_leave_out
from ensemble_utils import kensemble_validation
from train import df_preprocess
from ensemble_utils import generate_ensemblelist,kensemble_MAPtest
import hydra
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent  
class_dataloaders = { 'NRPS':[], 'other': [], 'PKS':[], 'ribosomal': [], 'saccharide': [], 'terpene': []}

if __name__ == "__main__":
    all_metrics = {}
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    with hydra.initialize(config_path =  os.path.join("..", "configs"), 
                          version_base="1.2"): 
        cfg = hydra.compose(config_name = "dataset", overrides=[f"BGC_MAP.device={device}"] )
    model_cfg = cfg.BGC_MAP
    BGC_data = pd.read_pickle(os.path.join(PROJECT_DIR, cfg.MAP_metadata))
    BGC_data = df_preprocess(cfg, model_cfg.task, BGC_data)
    for biosyn_class in class_dataloaders:
        class_dataloaders[biosyn_class]=generate_leave_out(BGC_data, 
                                                           biosyn_class, 
                                                           model_cfg.leave_out.test_ratio, 
                                                           model_cfg.data)
        print(biosyn_class, 
              len(class_dataloaders[biosyn_class][1]["train"].dataset),
              len(class_dataloaders[biosyn_class][1]["val"].dataset),
              len(class_dataloaders[biosyn_class][0].dataset))
    
    for biosyn_class in class_dataloaders:
        print(f"training_{biosyn_class}")
        kensemble_validation(class_dataloaders[biosyn_class], 
                             model_cfg, 
                             save_checkpoint = True, 
                             checkpoint_name = f"leave_out_{biosyn_class}")
    
    for biosyn_class in class_dataloaders:
        checkpoint_path = os.path.join(PROJECT_DIR, model_cfg.checkpoint_dir, f"leave_out_{biosyn_class}")
        ckpt = torch.load(os.path.join(checkpoint_path, f"leave_out_{biosyn_class}.ckpt"),weights_only = False)
        ensemble_model = generate_ensemblelist(ckpt)
        test_results = kensemble_MAPtest(ensemble_model, 
                                        class_dataloaders[biosyn_class][0], 
                                        checkpoint_path, 
                                        mean_result = True)
        all_metrics[biosyn_class] = test_results["metrics"]
        print(biosyn_class, test_results["metrics"])
    pickle.dump(all_metrics, open(os.path.join(PROJECT_DIR, model_cfg.leave_out.metric_path), "wb"))
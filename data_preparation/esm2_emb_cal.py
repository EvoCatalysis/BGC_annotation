import torch.nn as nn
import esm
import torch
import os
import pickle
import pandas as pd
from tqdm import tqdm
import hydra
from pathlib import Path
import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

PROJECT_DIR = Path(__file__).resolve().parent.parent

class Esm_BGC(object):
    def __init__(self, BGCs, sequence_strs):
        self.BGCs = list(BGCs)
        self.sequence_strs = list(sequence_strs)
        self.BGC_info=[]

        for i in range(len(self.BGCs)):
            for seq in self.sequence_strs[i]:
                self.BGC_info.append((self.BGCs[i],seq))

    @classmethod
    def from_df(cls, data):
        BGCs, sequence_strs = [], []

        BGCs = data["BGC_number"].tolist()
        sequence_strs = data["enzyme_list"].tolist()

        return cls(BGCs,sequence_strs)

    def __getitem__(self, idx):
        label,sequence=self.BGC_info[idx]
        return label,sequence

    def __len__(self):
        return len(self.BGC_info)
    
def generate_embedding(df, esm_ckpt):
    #model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_ckpt)
    dataset = Esm_BGC.from_df(df)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             collate_fn=alphabet.get_batch_converter(), 
                                             batch_size=32,
                                             shuffle=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        result={}
        for labels, strs, toks in tqdm(dataloader,desc="batch"):
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
                toks = toks[:,:1022]
            out=model(toks,repr_layers=[33],return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            for i,label in enumerate(labels): #len(labels)=batch_size
                result.setdefault(label, [])
                for layer,t in representations.items(): #t:batch_size x seq_length x hidden_dim
                    result[label].append(t[i,0,:].clone()) #first token
    return result

if __name__ == "__main__":

    with hydra.initialize(config_path=os.path.join("..", "configs"),
                          version_base="1.2"):
        cfg = hydra.compose(config_name="dataset")

    Esm2_model_path=os.path.join(PROJECT_DIR, "data","esm2_t33_650M_UR50D.pt")
    BGC_data=pd.read_pickle(os.path.join(PROJECT_DIR, cfg.MAC_metadata))
    #BGC_data=pd.read_pickle(os.path.join(PROJECT_DIR,"data", "toydata.pkl"))
    result = generate_embedding(BGC_data, Esm2_model_path)
    #torch.save(result, os.path.join(PROJECT_DIR,"data", "toydata_esm.pth"))
    #torch.save(result,os.path.join(PROJECT_DIR, cfg.ESM2_reps))


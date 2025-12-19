import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np
import pickle
from tqdm import tqdm
import os
import pandas as pd
import random
import argparse
import hydra
import re
from pathlib import Path
from BGC import Bgc
from rdkit import Chem
from data_utils import generate_negatives

PROJECT_DIR = Path(__file__).resolve().parent.parent


def canonize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol)
    return canonical_smiles

def generate_new_negatives(df, num_negatives = 2):
    positive_samples = df[df['is_product'] == 1]
    print("number of positive samples:", len(positive_samples))
    new_samples = []
    # create new negative for current positive
    for _, pos_row in tqdm(positive_samples.iterrows(), desc = "generating negatives from mibig"):
        current_bgc = pos_row['BGC_number']
        
        other_bgc_rows = positive_samples[positive_samples['BGC_number'] != current_bgc]
                    
        selected_negatives = other_bgc_rows.sample(n=num_negatives, replace=len(other_bgc_rows) < num_negatives)
        
        for _, neg_row in selected_negatives.iterrows():
            new_sample = {
                'BGC_number': current_bgc,
                'product': neg_row['product'],
                'biosyn_class': pos_row['biosyn_class'],
                'enzyme_list': pos_row['enzyme_list'],
                'is_product': 0
            }
            new_samples.append(new_sample)
    
    new_df = pd.concat([df, pd.DataFrame(new_samples)], ignore_index=True)
    
    return new_df


def list_files(directory): 
    file_paths = []  
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_paths.append(file_path)
    return file_paths

def extract_bgc_mac_dataset(json_dir: str, gbk_dir: str, output_dir: str) -> pd.DataFrame:
    """Extract BGC-MAC dataset from MIBiG JSON and GBK files."""
    # Get file paths
    json_paths = list_files(json_dir)
    gbk_paths = list_files(gbk_dir)
    # Filter JSON paths to match GBK files
    gbk_basenames = {os.path.splitext(os.path.basename(path))[0] for path in gbk_paths}
    filtered_json_paths = [path for path in json_paths if os.path.splitext(os.path.basename(path))[0] in gbk_basenames]
    gbk_paths = sorted(gbk_paths,
    key=lambda x: int(re.search(r"BGC(\d+)", x).group(1)) if re.search(r"BGC(\d+)", x) else 0
    )
    filtered_json_paths = sorted(filtered_json_paths,
    key=lambda x: int(re.search(r"BGC(\d+)", x).group(1)) if re.search(r"BGC(\d+)", x) else 0
    )

    # Process BGC data 
    columns = ["BGC_number", "product", "biosyn_class", "enzyme_list", "is_product"]
    BGC_data = pd.DataFrame(columns=columns)
    BGC_gene_kind = {}
    for gbk_file, json_file in tqdm(zip(gbk_paths, filtered_json_paths), desc="Processing BGC files"):
        assert os.path.splitext(os.path.basename(gbk_file))[0] == os.path.splitext(os.path.basename(json_file))[0], "BGC number doesn't match!"
        mibig_BGC = Bgc(gbk_file, json_file)  
        bgc_info = mibig_BGC.get_info()
        for info in bgc_info:
            BGC_data.loc[len(BGC_data)] = info

        # extract gene_kind
        try:
            mibig_BGC=Bgc(gbk_file,json_file)
            BGC_gene_kind[mibig_BGC.bgc_number]=mibig_BGC.get_gene_kind()
        except Exception as e:
            print(gbk_file,{e})
    
    with open(os.path.join(PROJECT_DIR,"data","BGC_4.0","BGC_gene_kind.pkl"), 'wb') as f:
        pickle.dump(BGC_gene_kind, f)


    # Filter out specific BGC
    BGC_data = BGC_data[BGC_data["BGC_number"] != "BGC0002977"].reset_index(drop=True)
    # Create MAC dataset
    MAC_dataset = (
        BGC_data.groupby("BGC_number")
        .agg({
            "biosyn_class": "first",  
            "enzyme_list": "first",  
        })
        .reset_index()
    )
    
    # Save MAC dataset
    os.makedirs(output_dir, exist_ok=True)
    MAC_dataset.to_pickle(os.path.join(output_dir, "MAC_metadata.pkl"))
    
    return BGC_data

def extract_bgc_map_dataset(BGC_data: pd.DataFrame, output_dir: str) -> tuple[pd.DataFrame, dict, dict, list]:
    """Extract BGC-MAP dataset (positive examples)."""
    # Clean data
    BGC_data_cleaned = BGC_data[BGC_data['product'] != '']
    BGC_data_cleaned = (
        BGC_data_cleaned.drop_duplicates(subset=["BGC_number", "product"])
        .reset_index(drop=True)
    )
    
    # Save cleaned data
    BGC_data_cleaned.to_pickle(os.path.join(output_dir, "BGC_structure4.pkl"))
    
    # Create dictionaries for later use
    smiles_group = BGC_data_cleaned.groupby("product")["BGC_number"].apply(list).to_dict()
    smiles_dict = BGC_data_cleaned.set_index("product")["BGC_number"].to_dict()
    mibig_np = list(smiles_dict.keys())
    
    return BGC_data_cleaned, smiles_group, smiles_dict, mibig_np

def convert_smiles_to_fingerprints(smiles_list: list[str]) -> list[tuple]:
    """Convert SMILES to molecular fingerprints."""
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    fps = []
    for smiles in tqdm(smiles_list, desc="Converting SMILES to fingerprints"):
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule:
                fps.append((morgan_generator.GetFingerprint(molecule), smiles)) # type: ignore
        except Exception as e:
            print(f"{smiles} error: {e}")
    return fps

def load_coconut_dataset(coconut_path: str, sample_ratio: float = 0.1) -> list[tuple]:
    """Load and sample molecules from COCONUT database."""
    coconut_mol = []    
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)

    # Count total lines
    with open(coconut_path, 'r') as f:
        total_lines = sum(1 for line in f)

    # Randomly select lines
    selected_lines = set(random.sample(range(total_lines), int(sample_ratio * total_lines)))
    
    # Load selected molecules
    with open(coconut_path, 'r') as f:
        for i, line in tqdm(enumerate(f), desc="Loading COCONUT molecules"):
            if i in selected_lines:
                smiles = line.split()[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    coconut_mol.append(mol)
    
    # Calculate fingerprints
    coconut_fps = []
    for mol in tqdm(coconut_mol, desc="Calculating COCONUT fingerprints"):
        try:
            coconut_fps.append((morgan_generator.GetFingerprint(mol), Chem.MolToSmiles(mol))) # type: ignore
        except Exception as e:
            print(f"Error calculating fingerprint: {e}")
            continue
            
    return coconut_fps

def calculate_similarity_matrix(fps1: list[tuple], fps2: list[tuple], output_path: str) -> np.ndarray:
    """Calculate similarity matrix between two sets of fingerprints."""
    similarity_matrix = np.zeros((len(fps1), len(fps2)))

    for i, fp1 in tqdm(enumerate(fps1), desc="Calculating similarity matrix"):
        for j, fp2 in enumerate(fps2):
            similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(fp1[0], fp2[0])

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Save similarity matrix
    with open(output_path, 'wb') as f:
        pickle.dump(similarity_matrix, f)
        
    return similarity_matrix

def create_map_dataset(BGC_data: pd.DataFrame, BGC_data_cleaned: pd.DataFrame, 
                      similarity_matrix: np.ndarray, smiles_group: dict, 
                      smiles_dict: dict, coconut_fps: list[tuple], 
                      output_dir: str) -> pd.DataFrame:
    """Create the final MAP dataset with positive and negative examples."""
    # Generate negatives based on similarity
    lower = np.percentile(similarity_matrix, 99.9, axis=1)
    higher = 0.8
    result = generate_negatives(lower, higher, 3, similarity_matrix, smiles_group, 
                               list(smiles_dict.keys()), coconut_fps, BGC_data_cleaned)
    
    # Add negative examples to dataset
    for dict_item in result:
        new_row_df = pd.DataFrame([dict_item])
        BGC_data_cleaned = pd.concat([BGC_data_cleaned, new_row_df], ignore_index=True)

    # Sample NPs for BGC without NP structures as negative data
    data_nostructure = BGC_data[BGC_data['product'] == '']
    data_nostructure = data_nostructure.drop_duplicates(subset="BGC_number")
    data_nostructure["is_product"] = 0
    
    # Sample 3 negatives from COCONUT
    sampled_NP = random.sample(coconut_fps, len(data_nostructure) * 4)
    random_sample = pd.concat([data_nostructure, data_nostructure.copy()], ignore_index=True)
    random_sample = pd.concat([random_sample, random_sample.copy()], ignore_index=True)
    random_sample["product"] = [NP[1] for NP in sampled_NP]
    
    # Combine datasets
    MAP_dataset = pd.concat([BGC_data_cleaned, random_sample])

    # Sample 2 negatives from mibig
    MAP_dataset = generate_new_negatives(MAP_dataset, num_negatives=2)

    # Canonize SMILES
    MAP_dataset["product"] = MAP_dataset["product"].apply(canonize_smiles)

    # Save metadata
    MAP_dataset.to_pickle(os.path.join(output_dir, "MAP_metadata.pkl"))
    
    return MAP_dataset

if __name__ == "__main__":
    with hydra.initialize(config_path =  os.path.join("..", "configs"),version_base="1.2"): 
        cfg = hydra.compose(config_name = "dataset")
    random.seed(cfg.seed)
    parser = argparse.ArgumentParser(description="Process BGC and natural product data")
    parser.add_argument("--json_dir", default=os.path.join("..", cfg.json_dir),
                        help="Directory containing MIBiG JSON files")
    parser.add_argument("--gbk_dir", default=os.path.join("..", cfg.gbk_dir),
                        help="Directory containing MIBiG GBK files")
    parser.add_argument("--output_dir", default=os.path.join("..", cfg.BGC_data_dir),
                        help="Directory to save output files")
    parser.add_argument("--np_dir", default=os.path.join("..", cfg.np_data_dir),
                        help="Directory to save output files")
    parser.add_argument("--coconut_path", 
                        default=os.path.join("..", cfg.np_data_dir, "COCONUT_DB_absoluteSMILES.smi"),
                        help="Path to COCONUT database file")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                        help="Ratio of COCONUT database to sample")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.np_dir, exist_ok=True)
    
    # Extract BGC-MAC dataset
    BGC_data = extract_bgc_mac_dataset(args.json_dir, args.gbk_dir, args.output_dir)
    print("Successfully create BGC-MAC metadata")
    # Extract BGC-MAP dataset (positive)
    BGC_data_cleaned, smiles_group, smiles_dict, mibig_np = extract_bgc_map_dataset(BGC_data, args.output_dir)
    
    # Convert SMILES to fingerprints
    mibig_fps = convert_smiles_to_fingerprints(mibig_np)
    
    # Load COCONUT dataset
    coconut_fps = load_coconut_dataset(args.coconut_path, args.sample_ratio)
    print("Successfully load coconut dataset")

    # Save fingerprints
    fps1_path = os.path.join(PROJECT_DIR, cfg.np_data_dir, "mibig_mol.pkl")
    fps2_path = os.path.join(PROJECT_DIR, cfg.np_data_dir, "coconut_mol.pkl")
    with open(fps1_path, 'wb') as f:
        pickle.dump(mibig_fps, f)
    with open(fps2_path, 'wb') as f:
        pickle.dump(coconut_fps, f)
    
    # Calculate similarity matrix
    similarity_path = os.path.join(PROJECT_DIR, cfg.np_data_dir, "similarity_matrix4.pkl")
    similarity_matrix = calculate_similarity_matrix(mibig_fps, coconut_fps, similarity_path)
    
    # Create MAP dataset
    MAP_dataset = create_map_dataset(BGC_data, BGC_data_cleaned, similarity_matrix, 
                                    smiles_group, smiles_dict, coconut_fps, args.output_dir)
    
    print("Data processing completed successfully!")




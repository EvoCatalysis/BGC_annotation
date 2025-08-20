
# **BGC-MAC and BGC-MAP: Attention-Based Models for Accurate Classification and Product Matching of Biosynthetic Gene Clusters**

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.11-blue.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-2.6-orange.svg" /></a>

</p>

## Setup

```bash
  conda env create -f environment.yml
  conda activate natural_product
  # Install local package.
  # Current directory should be natural_product/
  pip install -e .
```
## Reproducing Results
### Data Preprocessing (optional)
Training data and checkpoint files can be downloaded at [Zenodo](https://zenodo.org/records/15206672?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY0ZTg2YTA0LWJkN2UtNGQxYi05NmQxLTNkYTQwYzcyZjcyMSIsImRhdGEiOnt9LCJyYW5kb20iOiJhYzc1YjhlMzE1MWEyMTU2N2NmZTUzYzdhNDU0YzZkYyJ9.-GQz8GN03tDiiII3M06pFanYo9CTaPxEml1RhGIg2Ttgsir1cjX4FzaSKWy5WLuasc_sGozUmMafArCceWb1hw) .
1. Download MIBiG raw data
```bash
wget https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz
wget https://dl.secondarymetabolites.org/mibig/mibig_gbk_4.0.tar.gz
```

After downloading, extract the files and place them in the `./data` directory. Rename the directories to exactly:

- `mibig_gbk_4.0`
- `mibig_json_4.0`

2. Download pfam hmm file
```bash
wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
```

Extract the file and place them in the `./data` directory.
Ensure the path is `/data/Pfam-A.hmm`

3. Download ESM2 weights (see Usage Instructions-Prerequisites part)

4. Download `COCONUT_DB_absoluteSMILES.smi` 

5. Then run:

```bash
cd data_preparation
python extract_BGCs.py 
# Will generate three files: 
# ./data/BGC_4.0/MAC_metadata.pkl 
# ./data/BGC_4.0/BGC_gene_kind.pkl
# ./data/BGC_4.0/MAP_metadata.pkl
python pfam_annotation.py 
# Will generate one file: ./data/BGC_4.0/BGC_domain_pfam.pkl
python esm2_emb_cal.py 
# Will generate one file: ./data/BGC_4.0/Esm2_rep_mibig.pth
```

### Experiments
0. If you skip data preprocessing step, please download `BGC_4.0.zip` and place it in `./data` directory

    Before training, make sure `./data/BGC_4.0` has the following five files:
    - `BGC_domain_pfam.pkl`
    - `BGC_gene_kind.pkl`
    - `Esm2_rep_mibig.pth`
    - `MAC_metadata.pkl`
    - `MAP_metadata.pkl`

1. Download the following test files and place them in:
    - `./ckpt/BGC_MAC/test_MAC_metadata_43.pkl`
    - `./ckpt/BGC_MAP/test_MAP_metadata_42.pkl`
2. Execute experiments:

```bash
cd experiment

# Training
python train.py --model MAC
python train.py --model MAP

# Evaluation (replace 'your_ckpt_dir_name' with actual directory)
python test.py --model MAC --ckpt your_ckpt_dir_name --mean_result
python test.py --model MAP --ckpt your_ckpt_dir_name --mean_result

# To evaluate individual models in an ensemble (without mean results):
python test.py --model MAC --ckpt your_ckpt_dir_name 
python test.py --model MAP --ckpt your_ckpt_dir_name 

```

## Usage Instructions
### Prerequisites
1. Download pretrained ESM2 weights from [https://zenodo.org/records/7566741](https://zenodo.org/records/7566741)
    - [esm2_t33_650M_UR50D-contact-regression.pt](https://zenodo.org/records/7566741/files/esm2_t33_650M_UR50D-contact-regression.pt?download=1)
    - [esm2_t33_650M_UR50D.pt](https://zenodo.org/records/7566741/files/esm2_t33_650M_UR50D.pt?download=1)
    
    Place both files in `./data`
    
2. Place checkpoint files in:
    - `ckpt/BGC_MAC/MAC_default/MAC_default.ckpt`
    - `ckpt/BGC_MAP/MAP_default/MAP_default.ckpt`
    
    Make sure: The filename stem (without extension) of a .ckpt file must match the name of its parent directory

### Input Format Specifications
- **Single BGC classification**: Directly provide GBK file path
- **Multiple BGC classification**: Provide directory path (automatically processes all GBK files)
    - GBK files must follow MIBiG or antiSMASH format
- **Single product matching**: Provide SMILES string directly
- **Multiple product matching**: Provide a pickle file 
containing:
    - A list with length matching the number of GBK files
    - Each element is a sublist representing products to match for that BGC
- **BGC_ranking**: Provide the directory path for all candidate BGC and a SMILES string.


Example format for 4 BGCs:

```python
smiles = [
    ["CCO", "C=O"],          
    ["C1=CC=CC=C1", "CCN"],  
    ["O=C(O)C", "C#N"],    
    ["CC(C)=O", "CCl"]  
]
```

### Prediction Commands

```bash
cd experiment

#Multiple BGC classification
python predict_new.py --gbk ../data/example/BGC0000001.gbk

#Single BGC classification
python predict_new.py --gbk ../data/example

#Single product matching
python predict_new.py --gbk ../data/example/BGC0001178.gbk --smiles "O=C1N[C@@H](C2=CC(O3)=CC(OS(O)(=O)=O)=C2)C(N[C@@H](C(N[C@@H]45)=O)C6=CC(OC7=C(Cl)C=C(C[C@@H]1NC(C(C8=CC3=C(O)C(Cl)=C8)=O)=O)C=C7)=C(O[C@@H]9[C@H](OC%10O[C@@H](C)[C@@H](O)[C@](N)(C)C%10)[C@@H](O)[C@H](O)[C@@H](CO)O9)C(OC%11=CC=C([C@@H](O)[C@H](NC4=O)C(N[C@@H](C(O)=O)C%12=CC(O)=CC(O)=C%12C%13=CC5=CC(Cl)=C%13O)=O)C=C%11Cl)=C6)=O"

#Multiple product matching
python predict_new.py --gbk ../data/example --smiles ../data/smiles.pkl

#BGC ranking
python predict_new.py --gbk ../data/example --smiles "CN(C([C@H](CC1=CC=C(OC)C=C1)N(C)C(/C=C(C)/OC)=O)=O)[C@@H](CC(C)C)C(N(C)[C@@H]([C@H](CC)C)C(N2CCC[C@H]2C3=NC=CS3)=O)=O"

```

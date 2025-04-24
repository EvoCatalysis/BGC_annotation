
# **BGC-MAC and BGC-MAP: Attention-Based Models for Accurate Classification and Product Matching of Biosynthetic Gene Clusters**

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

```bash
wget https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz
wget https://dl.secondarymetabolites.org/mibig/mibig_gbk_4.0.tar.gz
```

After downloading, extract the files and place them in the `./data` directory. Rename the directories to exactly:

- `mibig_gbk_4.0`
- `mibig_json_4.0`

Then run:

```bash
cd data_preparation
python extract_BGCs.py
```

### Experiments

1. Download the following test files and place them in:
    - `./ckpt/BGC_MAC/test_classification_43.pkl`
    - `./ckpt/BGC_MAP/test_product_matching_42.pkl`
2. Execute experiments:

```bash
cd experiment

# Training
python train.py --model MAC
python train.py --model MAP

# Generalization analysis
python generalization.py

# Evaluation (replace 'your_ckpt_dir_name' with actual directory)
python test.py --model MAC --ckpt your_ckpt_dir_name --mean_result
python test.py --model MAC --ckpt your_ckpt_dir_name --mean_result

# To evaluate individual models in an ensemble (without mean results):
python test.py --model MAC --ckpt your_ckpt_dir_name 
python test.py --model MAC --ckpt your_ckpt_dir_name 

```

## Usage Instructions
### Prerequisites
1. Download pretrained ESM2 weights from [https://zenodo.org/records/7566741](https://zenodo.org/records/7566741)
    - [esm2_t33_650M_UR50D-contact-regression.pt](https://zenodo.org/records/7566741/files/esm2_t33_650M_UR50D-contact-regression.pt?download=1)
    - [esm2_t33_650M_UR50D.pt](https://zenodo.org/records/7566741/files/esm2_t33_650M_UR50D.pt?download=1)
    
    Place both files in `./data`
    
2. Place checkpoint files in:
    - `ckpt/BGC_MAC/default/default.ckpt`
    - `ckpt/BGC_MAP/default/default.ckpt`
### Input Format Specifications
- **Single BGC classification**: Directly provide GBK file path
- **Multiple BGC classification**: Provide directory path (automatically processes all GBK files)
    - GBK files must follow MIBiG or antiSMASH format
- **Single product matching**: Provide SMILES string directly
- **Multiple product matching**: Provide a pickle file containing:
    - A list with length matching the number of GBK files
    - Each element is a sublist representing products to match for that BGC

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

#Single BGC classification
python predict_new.py --gbk ../data/example

#Multiple BGC classification
python predict_new.py --gbk ../data/example/BGC0000001.gbk

#Single product matching
python predict_new.py --gbk ../data/example/BGC0001178.gbk --smiles "O=C1N[C@@H](C2=CC(O3)=CC(OS(O)(=O)=O)=C2)C(N[C@@H](C(N[C@@H]45)=O)C6=CC(OC7=C(Cl)C=C(C[C@@H]1NC(C(C8=CC3=C(O)C(Cl)=C8)=O)=O)C=C7)=C(O[C@@H]9[C@H](OC%10O[C@@H](C)[C@@H](O)[C@](N)(C)C%10)[C@@H](O)[C@H](O)[C@@H](CO)O9)C(OC%11=CC=C([C@@H](O)[C@H](NC4=O)C(N[C@@H](C(O)=O)C%12=CC(O)=CC(O)=C%12C%13=CC5=CC(Cl)=C%13O)=O)C=C%11Cl)=C6)=O"

#Multiple product matching
python predict_new.py --gbk ../data/example --smiles ../data/smiles.pkl
```

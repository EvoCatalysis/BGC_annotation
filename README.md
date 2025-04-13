## Installation
```bash
  conda env create -f environment.yml
  conda activate natural_product
  # Install local package.
  # Current directory should be natural_product/
  pip install -e .
```
## Training

## Inference new BGCs
```bash
  #classification
  python predict_new.py --gbk ../data/example/new
  python predict_new.py --gbk ../data/example/new/BGC0000001.gbk
  #product matching
  python predict_new.py --gbk ../data/example/new/BGC0001178.gbk --model MAP --ckpt MAP_ckpt1 --smiles "O=C1N[C@@H](C2=CC(O3)=CC(OS(O)(=O)=O)=C2)C(N[C@@H](C(N[C@@H]45)=O)C6=CC(OC7=C(Cl)C=C(C[C@@H]1NC(C(C8=CC3=C(O)C(Cl)=C8)=O)=O)C=C7)=C(O[C@@H]9[C@H](OC%10O[C@@H](C)[C@@H](O)[C@](N)(C)C%10)[C@@H](O)[C@H](O)[C@@H](CO)O9)C(OC%11=CC=C([C@@H](O)[C@H](NC4=O)C(N[C@@H](C(O)=O)C%12=CC(O)=CC(O)=C%12C%13=CC5=CC(Cl)=C%13O)=O)C=C%11Cl)=C6)=O"
```

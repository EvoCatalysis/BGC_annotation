import torch

class BaseDataset(object):
    def __init__(self, BGCs, biosyn_class, pros, gene_kind, pfam, structure=None):
        self.BGCs = list(BGCs)
        self.length = [len(x) for x in pros] #BGC_length 
        self.pros = list(pros)
        self.biosyn_class = list(biosyn_class)
        self.gene_kind = list(gene_kind) if gene_kind is not None else None
        self.pfam = list(pfam) if pfam is not None else None
        self.structure = list(structure) if structure is not None else None

    @classmethod
    def from_df(cls, data, use_structure):
        BGCs = data["BGC_number"].tolist()
        biosyn_class = data["biosyn_class"].tolist()
        pros = data["protein_rep"].tolist()
        gene_kind = data["gene_kind"].tolist() if "gene_kind" in data else None
        pfam = data["pfam"].tolist() if "pfam" in data else None

        structure = None
        if use_structure:
            if 'structure' not in data.columns:
                raise ValueError("config must be provided when use_structure is True")
            structure = data["structure"].tolist()

        return cls._create_instance(BGCs, biosyn_class, pros, gene_kind, pfam, structure, data)
    
    @classmethod
    def _create_instance(cls, BGCs, biosyn_class, pros, gene_kind, pfam, structure, data):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        return len(self.BGCs)
    
    def _get_common_items(self, idx):
        """Get items common to both dataset types"""
        structure = self.structure[idx] if self.structure is not None else None
        gene_kind = self.gene_kind[idx] if self.gene_kind is not None else None
        pfam = self.pfam[idx] if self.pfam is not None else None
        label = self.BGCs[idx]
        biosyn_class = self.biosyn_class[idx]
        pro = self.pros[idx]
        length = self.length[idx]
        
        return label, biosyn_class, pro, length, structure, gene_kind, pfam

class MACDataset(BaseDataset):
    def __init__(self, BGCs, biosyn_class, pros, gene_kind, pfam, structure=None):
        super().__init__(BGCs, biosyn_class, pros, gene_kind, pfam, structure)
        self.class_token = torch.tensor([0, 1, 2, 3, 4, 5])

    @classmethod
    def _create_instance(cls, BGCs, biosyn_class, pros, gene_kind, pfam, structure, data):
        return cls(BGCs, biosyn_class, pros, gene_kind, pfam, structure)

    def __getitem__(self, idx):
        label, biosyn_class, pro, length, structure, gene_kind, pfam = self._get_common_items(idx)
        class_token = self.class_token
        return label, biosyn_class, pro, length, structure, class_token, gene_kind, pfam

class MAPDataset(BaseDataset):
    def __init__(self, BGCs, biosyn_class, pros, subs, is_product, gene_kind, pfam, structure=None):
        super().__init__(BGCs, biosyn_class, pros, gene_kind, pfam, structure)
        self.subs = list(subs)
        self.is_product = list(is_product)

    @classmethod
    def _create_instance(cls, BGCs, biosyn_class, pros, gene_kind, pfam, structure, data):
        subs = data["product_index"].tolist()
        is_product = data["is_product"].tolist()
        return cls(BGCs, biosyn_class, pros, subs, is_product, gene_kind, pfam, structure)

    def __getitem__(self, idx):
        label, biosyn_class, pro, length, structure, gene_kind, pfam = self._get_common_items(idx)
        sub = self.subs[idx]
        is_product = self.is_product[idx]
        return label, biosyn_class, pro, sub, is_product, length, structure, gene_kind, pfam

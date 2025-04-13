from Bio import SeqIO
import os
class Natural_product:
    def __init__(self,biosyn_class,name,smile,chem_act = None):
        self.name = name
        self.biosyn_class = biosyn_class
        self.smile = smile
        self.chem_act = chem_act
    def __str__(self):
        return str(self.name)
    
class Enzyme:
    def __init__(self,identifier,nu_seq,aa_seq,gene_functions=None,gene_kind=None): #identifer is a dictionary
        self.identifier=identifier
        self.nu_sequence=nu_seq
        self.aa_sequence=aa_seq
        self.gene_functions=gene_functions
        self.gene_kind=gene_kind
        self.domain=[]
    def set_domain(self,name,label,locus_tag,nu_seq,aa_seq):
        self.domain.append(Domain(name,label,locus_tag,nu_seq,aa_seq))
    def domain_checker(self):
        if self.domain:
            for domains in self.domain:
                if domains.aa_sequence in self.aa_sequence:
                    continue
                else:
                    print(domains)
                    return False
        return True
    def __str__(self):
        output = ','.join([f'{key}:{value}' for key, value in self.identifier.items()])
        return output
    
class Domain:
    def __init__(self,name,label,locus_tag,nu_seq,aa_seq):
        self.name=name
        self.label=label
        self.locus_tag=locus_tag
        self.nu_sequence=nu_seq
        self.aa_sequence=aa_seq
    def __str__(self):
        return self.label

class Bgc:
    def __init__(self, gbk, json = None, database="mibig", product=None):
        filename = os.path.basename(gbk)
        self.database=database
        self.enzyme_list=self.get_enzyme(gbk) 
        if database=="mibig":
            if json is None:
                raise ValueError("Please provide a JSON file for MIBiG database.")
            self.product=self.get_product(json) 
            self.bgc_number = filename.split(".")[0]
        elif database=="antismash":
            self.bgc_number = ".".join(filename.split(".")[:3])
        elif database=="new":
            self.bgc_number = os.path.basename(gbk).split(".")[0]
            if product is not None:
                self.product = [Natural_product(None, None, smile) for smile in product]
            else:
                self.product = [Natural_product(None, None, "")]

    def get_product(self,filename): 
        import json
        result=[]
        biosyn_class = []
        with open(filename, "r") as file:
            json_text = file.read()
        data = json.loads(json_text)
        compound_types = data["biosynthesis"]["classes"]
        for compound_type in compound_types:
            biosyn_class.append(compound_type["class"])
        compounds = data["compounds"]
        for compound in compounds:
            chem_acts = []
            compound_name = compound["name"]
            try:
                chem_struct = compound["structure"]
            except:
                chem_struct=""
            try:
                for item in compound["bioactivities"]:
                    chem_acts.append(item["name"])
            except:
                pass  
            result.append(Natural_product(biosyn_class, compound_name, chem_struct, chem_acts))
        return result
    
    def get_enzyme(self,gbk_file): #extract enzyme information from gbk file
        records = SeqIO.parse(gbk_file, "genbank")
        enzyme_list=[]
        first_domain=[]
        for record in records:
            if self.database=="antismash":
                try:
                    self.product=record.features[0].qualifiers.get("product")[0]
                except:
                    self.product="error"
            for feature in record.features:
                if feature.type=="CDS":
                    identifier={}
                    gene_functions,gene_kind=None,None
                    if feature.qualifiers.get("protein_id"):
                        identifier["protein_id"]=feature.qualifiers.get("protein_id")[0]
                    if feature.qualifiers.get("gene"):
                        identifier["gene"]=feature.qualifiers.get("gene")[0]
                    if feature.qualifiers.get("locus_tag"):
                        identifier["locus_tag"]=feature.qualifiers.get("locus_tag")[0]
                    if feature.qualifiers.get("gene_functions"):
                        gene_functions=feature.qualifiers.get("gene_functions")[0]
                    if feature.qualifiers.get("gene_kind"):
                        gene_kind=feature.qualifiers.get("gene_kind")[0]
                    nu_sequence=feature.extract(record).seq
                    aa_sequence=feature.qualifiers.get("translation")[0]
                    enzyme_list.append(Enzyme(identifier,nu_sequence,aa_sequence,gene_functions,gene_kind))
                    if len(first_domain)==5: #some domains occurr before the CDS occurrs
                        enzyme_list[0].set_domain(*first_domain)
                        first_domain=[]
                if feature.type=="aSDomain": #if the enzyme has domains, record its domain information
                    nu_sequence=feature.extract(record).seq
                    aa_sequence=feature.qualifiers.get("translation")[0]
                    if self.database=="antismash":
                        name,locus_tag=feature.qualifiers.get("asDomain_id")[0],feature.qualifiers.get("locus_tag")[0]
                        label=name
                    else:
                        name,label,locus_tag=feature.qualifiers.get("aSDomain")[0],feature.qualifiers.get("label")[0],feature.qualifiers.get("locus_tag")[0]
                    if enzyme_list:
                        enzyme_list[-1].set_domain(name,label,locus_tag,nu_sequence,aa_sequence)
                    else:
                        first_domain=[name,label,locus_tag,nu_sequence,aa_sequence]
        return enzyme_list
    def get_enzyme_list(self):
        enzyme_list=[]
        for enzymes in self.enzyme_list:
            if not enzymes.domain:
                enzyme_list.append(enzymes)
            else:   
                for domains in enzymes.domain:
                    enzyme_list.append(domains)
        return enzyme_list
    
    def get_chem_acts(self):
        chem_act=[]
        for products in self.product:
            chem_act.append(products.chem_acts)
        return list(set(chem_act))
    
    def get_gene_kind(self):
        gene_kind=[]
        for enzymes in self.enzyme_list:
            if not enzymes.domain or len(enzymes.domain)==1:
                gene_kind.append(enzymes.gene_kind)
            else:
                for domains in enzymes.domain:
                    gene_kind.append(enzymes.gene_kind)
        return gene_kind
    def get_info(self):
        enzyme_list=[pro.aa_sequence for pro in self.get_enzyme_list()]
        if self.database=="mibig":
            info = [[self.bgc_number,x.smile,x.biosyn_class,enzyme_list,1] for x in self.product]
        elif self.database=="new":
            info = [[self.bgc_number, x.smile, None, enzyme_list, 0] for x in self.product]
        else:
            info = [[self.bgc_number,0,self.product,enzyme_list,0]] 
        return info
    
    def __str__(self):
        return self.bgc_number

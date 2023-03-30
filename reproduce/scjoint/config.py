import torch
import os

class Config(object):
    def __init__(self):
        DB = 'HumanFetal'
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        
        if DB == '10x':
            # DB info
            self.number_of_class = 11
            self.input_size = 15463
            self.rna_paths = ['data_10x/exprs_10xPBMC_rna.npz']
            self.rna_labels = ['data_10x/cellType_10xPBMC_rna.txt']     
            self.atac_paths = ['data_10x/exprs_10xPBMC_atac.npz']
            self.atac_labels = [] #Optional. If atac_labels are provided, accuracy after knn would be provided.
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = ''
        
        elif DB == "MOp":
            self.number_of_class = 21
            self.input_size = 18603
            self.rna_paths = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz',\
                                'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
            self.rna_labels = ['data_MOp/YaoEtAl_RNA_snRNA_10X_v3_B_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt',\
                                'data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
            self.atac_paths = ['data_MOp/YaoEtAl_ATAC_exprs.npz',\
                                'data_MOp/YaoEtAl_snmC_exprs.npz']
            self.atac_labels = ['data_MOp/YaoEtAl_ATAC_cellTypes.txt',\
                                'data_MOp/YaoEtAl_snmC_cellTypes.txt']
            self.rna_protein_paths = []
            self.atac_protein_paths = []
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 20
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
            
        elif DB == "db4_control":
            self.number_of_class = 7 # 7 # Number of cell types in CITE-seq data
            self.input_size = 17668 # 17668 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            self.rna_paths = ['data/CITE-ASAP/citeseq_control_rna.npz'] # RNA gene expression from CITE-seq data
            self.rna_labels = ['data/CITE-ASAP/citeseq_control_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.atac_paths = ['data/CITE-ASAP/asapseq_control_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['data/CITE-ASAP/asapseq_control_cellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = ['data/CITE-ASAP/citeseq_control_adt.npz'] # Protein expression from CITE-seq data
            self.atac_protein_paths = ['data/CITE-ASAP/asapseq_control_adt.npz'] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20
            self.epochs_stage3 = 20
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 

        elif DB == "MCA_subset":
            self.number_of_class = 19 # Number of cell types in CITE-seq data
            self.input_size = 15519 # Number of common genes and proteins between CITE-seq data and ASAP-seq
#             self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
#                               '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
#             self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
#                                '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz']
            self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt']
            
            self.atac_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_atac_CellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.001
            self.lr_stage3 = 0.001
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 20  # ep1=20, ep2=10, w=1, acc=0.827 | ep1=ep2=10, w=1, acc=0.820
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
        
        elif DB == "MCA":
            self.number_of_class = 73 # Number of cell types in CITE-seq data
            self.input_size = 15519 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_rna_facs.npz', 
                              '/home/yxh/data/MCA/scjoint/data_atlas/atlas_rna_dropseq.npz']
            self.rna_labels = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_rna_facs_cellTypes.txt',
                               '/home/yxh/data/MCA/scjoint/data_atlas/atlas_rna_dropseq_cellTypes.txt']
            
            self.atac_paths = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_atac_cellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
            
        elif DB == "MCA_facs":
            self.number_of_class = 67 # Number of cell types in CITE-seq data
            self.input_size = 15519 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_rna_facs.npz']
            self.rna_labels = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_rna_facs_CellTypes.txt']
            
            self.atac_paths = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/MCA/scjoint/data_atlas/atlas_atac_CellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 

        elif DB == 'MCA_OS':
            self.number_of_class = 19 # Number of cell types in CITE-seq data
            self.input_size = 15519 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths = ['/home/yxh/data/MCA/scjoint/data_os/subset_rna_facs.npz']
            self.rna_labels = ['/home/yxh/data/MCA/scjoint/data_os/atlas_rna_facs_cellTypes.txt']
            
            self.atac_paths = ['/home/yxh/data/MCA/scjoint/data_os/atlas_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/MCA/scjoint/data_os/atlas_atac_CellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 

        elif DB == 'PBMCMultome':
            self.number_of_class = 19 # Number of cell types in CITE-seq data
            self.input_size = 18353 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths = ['/home/yxh/gitrepo/multi-omics/scJoint-main/data/pbmc_10x/scjoint/rna.npz']
            self.rna_labels = ['/home/yxh/gitrepo/multi-omics/scJoint-main/data/pbmc_10x/scjoint/rna_CellTypes.txt']
            
            self.atac_paths = ['/home/yxh/gitrepo/multi-omics/scJoint-main/data/pbmc_10x/scjoint/atac_gam.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/gitrepo/multi-omics/scJoint-main/data/pbmc_10x/scjoint/atac_gam_CellTypes.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 50
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 1
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 

        elif DB == 'HumanFetal_50k':
            self.number_of_class = 54 # Number of cell types in CITE-seq data
            self.input_size = 22121 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths  = ['/home/yxh/data/HumanFetal_50k/RNA/adata_rna_sampled.npz']
            self.rna_labels = ['/home/yxh/data/HumanFetal_50k/RNA/rna_cell_types.txt']
            
            self.atac_paths  = ['/home/yxh/data/HumanFetal_50k/ATAC/adata_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/HumanFetal_50k/ATAC/atac_cell_types.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 256
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 

        elif DB == 'HumanFetal_100k':
            self.number_of_class = 54 # Number of cell types in CITE-seq data
            self.input_size = 22121 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths  = ['/home/yxh/data/HumanFetal_100k/RNA/adata_rna_sampled.npz']
            self.rna_labels = ['/home/yxh/data/HumanFetal_100k/RNA/rna_cell_types.txt']
            
            self.atac_paths  = ['/home/yxh/data/HumanFetal_100k/ATAC/adata_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/HumanFetal_100k/ATAC/atac_cell_types.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 512
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
            
        elif DB == 'HumanFetal_200k':
            self.number_of_class = 54 # Number of cell types in CITE-seq data
            self.input_size = 22121 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths  = ['/home/yxh/data/HumanFetal_200k/RNA/adata_rna_sampled.npz']
            self.rna_labels = ['/home/yxh/data/HumanFetal_200k/RNA/rna_cell_types.txt']
            
            self.atac_paths  = ['/home/yxh/data/HumanFetal_200k/ATAC/adata_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/HumanFetal_200k/ATAC/atac_cell_types.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 512
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
            
        elif DB == 'HumanFetal_400k':
            self.number_of_class = 54 # Number of cell types in CITE-seq data
            self.input_size = 22121 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths  = ['/home/yxh/data/HumanFetal_400k/RNA/adata_rna_sampled.npz']
            self.rna_labels = ['/home/yxh/data/HumanFetal_400k/RNA/rna_cell_types.txt']
            
            self.atac_paths  = ['/home/yxh/data/HumanFetal_400k/ATAC/adata_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/HumanFetal_400k/ATAC/atac_cell_types.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 512
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.8
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 
        
        elif DB == 'HumanFetal':
            self.number_of_class = 54 # Number of cell types in CITE-seq data
            self.input_size = 22121 # Number of common genes and proteins between CITE-seq data and ASAP-seq
            # self.rna_paths = ['/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_facs.npz',  
            #                   '/home/yxh/data/MCA/scjoint/atlas_subset/subset_rna_drop.npz'] # RNA gene expression from CITE-seq data
            # self.rna_labels = ['/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_facs_cellTypes.txt', 
            #                    '/home/yxh/data/MCA/scjoint/atlas_subset/atlas_rna_dropseq_cellTypes.txt'] # CITE-seq data cell type labels (coverted to numeric) 
            self.rna_paths  = ['/home/yxh/data/HumanFetal/RNA/adata_rna_sampled.npz']
            self.rna_labels = ['/home/yxh/data/HumanFetal/RNA/rna_cell_types.txt']
            
            self.atac_paths  = ['/home/yxh/data/HumanFetal/ATAC/adata_atac.npz'] # ATAC gene activity matrix from ASAP-seq data
            self.atac_labels = ['/home/yxh/data/HumanFetal/ATAC/atac_cell_types.txt'] # ASAP-seq data cell type labels (coverted to numeric) 
            self.rna_protein_paths = [] # Protein expression from CITE-seq data
            self.atac_protein_paths = [] # Protein expression from ASAP-seq data
            
            # Training config            
            self.batch_size = 1024
            self.lr_stage1 = 0.01
            self.lr_stage3 = 0.01
            self.lr_decay_epoch = 20
            self.epochs_stage1 = 10
            self.epochs_stage3 = 10
            self.p = 0.4
            self.embedding_size = 64
            self.momentum = 0.9
            self.center_weight = 10
            self.with_crossentorpy = True
            self.seed = 1
            self.checkpoint = '' 




            

        




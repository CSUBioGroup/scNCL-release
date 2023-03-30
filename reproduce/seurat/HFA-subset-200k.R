library(Seurat)
library(glue)
# ================================================
# HumanFetal 200k dataset
# ================================================
library(Matrix)
read_dir = glue("/home/yxh/data/HumanFetal_200k/")

b1_exprs_filename = "adata_rna_sampled.mtx"
b2_exprs_filename = "adata_atac.mtx"
b1_cnames_filename = 'rna_cnames.csv'
b2_cnames_filename = 'atac_cnames.csv'
b1_celltype_filename = "rna_cell_types.csv"
b2_celltype_filename = "atac_cell_types.csv"
gnames_filename    = 'gnames.csv'


########################
# read data 

b1_exprs <- readMM(file = paste0(read_dir, 'RNA/', b1_exprs_filename))
b2_exprs <- readMM(file = paste0(read_dir, 'ATAC/', b2_exprs_filename))
b1_meta <- read.table(file = paste0(read_dir, 'RNA/', b1_celltype_filename),sep=",",header=T,row.names=1,check.names = F)
b2_meta <- read.table(file = paste0(read_dir, 'ATAC/', b2_celltype_filename),sep=",",header=T,row.names=1,check.names = F)

b1_cnames = read.table(file=paste0(read_dir, 'RNA/', b1_cnames_filename), sep=',', header=T) 
b2_cnames = read.table(file=paste0(read_dir, 'ATAC/', b2_cnames_filename), sep=',', header=T)
gnames = read.table(file=paste0(read_dir, 'RNA/', gnames_filename), header=T, sep=',')
# head(gnames)

b1_exprs = as.matrix(t(b1_exprs))
b2_exprs = as.matrix(t(b2_exprs))
rownames(b1_exprs) = gnames$X0
colnames(b1_exprs) = b1_cnames$X0
rownames(b2_exprs) = gnames$X0
colnames(b2_exprs) = b2_cnames$X0

label1 = b1_meta$Main_cluster_name
label2 = b2_meta$cell_type

rna.obj = CreateSeuratObject(counts=b1_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
rna.obj@meta.data[['cell_type']] = label1
rna.obj = NormalizeData(rna.obj, verbose = FALSE)
rna.obj = FindVariableFeatures(rna.obj, selection.method = "vst", nfeatures = 4000,
                verbose = FALSE)
rna.obj = ScaleData(rna.obj, verbose=FALSE)
rna.obj = RunPCA(rna.obj, npcs = 30, verbose = FALSE)

# atac objects 2
# norm and scale for findingAnchors
atac.obj = CreateSeuratObject(counts=b2_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
atac.obj = NormalizeData(atac.obj, verbose = FALSE)
atac.obj = ScaleData(atac.obj, verbose=FALSE)
atac.obj = RunPCA(atac.obj, features=VariableFeatures(object = rna.obj), npcs = 30, verbose = FALSE)


anchors <- FindTransferAnchors(reference = rna.obj, query = atac.obj,
    features = VariableFeatures(object = rna.obj),
    reduction='cca')

atac.pred <- TransferData(anchorset = anchors, refdata = rna.obj$cell_type,
    weight.reduction = atac.obj[["pca"]], dims = 1:30)

atac.obj <- AddMetaData(atac.obj, metadata = atac.pred)

atac.obj@meta.data[['cell_type']] = label2

n_corr = length(which(atac.obj$predicted.id == atac.obj$cell_type))
n_incorr = length(atac.obj$cell_type) - n_corr

n_corr / (n_corr + n_incorr)

# dir.create('/home/yxh/gitrepo/multi-omics/seurat/outputs')
write.csv(atac.pred, file='/home/yxh/gitrepo/multi-omics/seurat/outputs/HumanFetal_200k.csv', quote=F, row.names=T)

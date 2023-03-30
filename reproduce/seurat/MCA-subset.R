library(Seurat)
library(glue)
# ================================================
# MCAsubset dataset
# ================================================
library(Matrix)
read_dir = glue("/home/yxh/data/MCA/scjoint/atlas_subset/")

b1_exprs_filename = "subset_rna_facs.mtx"
b2_exprs_filename = "subset_atac.mtx"
b1_cnames_filename = 'subset_rna_facs_cnames.csv'
b1_celltype_filename = "atlas_rna_facs_cellTypes.csv"
b2_celltype_filename = "subset_atac_CellTypes.csv"
b2_cnames_filename = 'subset_atac_cnames.csv'
gnames_filename    = 'gnames.csv'


########################
# read data 

b1_exprs <- readMM(file = paste0(read_dir, b1_exprs_filename))
b2_exprs <- readMM(file = paste0(read_dir, b2_exprs_filename))
b1_meta <- read.table(file = paste0(read_dir, b1_celltype_filename),sep=",",header=T,row.names=1,check.names = F)
b2_meta <- read.table(file = paste0(read_dir, b2_celltype_filename),sep=",",header=T,row.names=1,check.names = F)

b1_cnames = read.table(file=paste0(read_dir, b1_cnames_filename), sep=',', header=T) 
b2_cnames = read.table(file=paste0(read_dir, b2_cnames_filename), sep=',', header=T)
gnames = read.table(file=paste0(read_dir, gnames_filename), header=T)

b1_exprs = as.matrix(t(b1_exprs))
b2_exprs = as.matrix(t(b2_exprs))
rownames(b1_exprs) = gnames$X0
colnames(b1_exprs) = b1_cnames$X0
rownames(b2_exprs) = gnames$X0
colnames(b2_exprs) = b2_cnames$X0

label1 = data.frame(type=b1_meta$x)
label2 = data.frame(type=b2_meta$cell_label)


rna.obj = CreateSeuratObject(counts=b1_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
rna.obj@meta.data[['cell_type']] = b1_meta$x 
# rna.x already log normalized
# rna.obj = NormalizeData(rna.obj, verbose = FALSE)
rna.obj = FindVariableFeatures(rna.obj, selection.method = "vst", nfeatures = 2000,
        		verbose = FALSE)
rna.obj = ScaleData(rna.obj, verbose=FALSE)
rna.obj = RunPCA(rna.obj, npcs = 30, verbose = FALSE)

# atac 
atac.obj = CreateSeuratObject(counts=b2_exprs, project = "rna", assay = "rna",
                                  min.cells = 0, min.features = 0,
                                  names.field = 1)
atac.obj = NormalizeData(atac.obj, verbose = FALSE)
atac.obj = ScaleData(atac.obj, verbose=FALSE)
atac.obj = RunPCA(atac.obj, features=VariableFeatures(object = rna.obj), npcs = 30, verbose = FALSE)

# check values, @data
# max(GetAssayData(object = mcasubset.atac))

anchors <- FindTransferAnchors(reference = rna.obj, query = atac.obj,
    features = VariableFeatures(object = rna.obj),
    reduction='cca')

#### Transfer annotations
atac.pred <- TransferData(anchorset = anchors, refdata = rna.obj$cell_type,
    weight.reduction = atac.obj[["pca"]], dims = 1:30)

atac.obj <- AddMetaData(atac.obj, metadata = atac.pred)
atac.obj@meta.data[['cell_type']] = b2_meta$cell_label

#### Impute GAM of scATAC-seq
genes.use <- VariableFeatures(rna.obj)
refdata <- GetAssayData(rna.obj, assay = "rna", slot = "data")[genes.use, ]

# refdata (input) contains a scRNA-seq expression matrix for the scRNA-seq cells.  imputation
# (output) will contain an imputed scRNA-seq matrix for each of the ATAC cells
imputation <- TransferData(anchorset = anchors, refdata = refdata, weight.reduction = atac.obj[["pca"]],
    dims = 1:30)
atac.obj[["rna"]] <- imputation

atac.obj <- ScaleData(atac.obj, features = genes.use, do.scale = FALSE)
atac.obj <- RunPCA(atac.obj, features = genes.use, verbose = FALSE)
atac.obj <- RunUMAP(atac.obj, dims = 1:30)
atac.obj.umap = Embeddings(atac.obj, reduction='umap')

n_corr = length(which(atac.obj$predicted.id == atac.obj$cell_type))
n_incorr = length(atac.obj$cell_type) - n_corr

n_corr / (n_corr + n_incorr)

dir.create('/home/yxh/gitrepo/multi-omics/seurat/outputs/')
write.csv(atac.pred, file='/home/yxh/gitrepo/multi-omics/seurat/outputs/MCAsubset.csv', quote=F, row.names=T)
Matrix::writeMM(obj = as(atac.obj.umap, 'sparseMatrix'), file="/home/yxh/gitrepo/multi-omics/seurat/outputs/mcasubset-atac-umap.mtx")

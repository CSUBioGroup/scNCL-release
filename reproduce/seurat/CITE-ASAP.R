library(Seurat)
library(glue)
library(Matrix)

read_dir = glue("/home/yxh/gitrepo/multi-omics/scJoint-main/data/CITE-ASAP/")

b1_exprs_filename = "citeseq_expr.mtx" # not normalized
b2_exprs_filename = "asapseq_expr.mtx"
b1_adt_filename   = 'citeseq_adt.mtx'  # lognormalized
b2_adt_filename   = 'asapseq_adt.mtx'

b1_cnames_filename = 'citeseq_cnames.csv'
b1_celltype_filename = "citeseq_control_cellTypes.csv"
b2_celltype_filename = "asapseq_control_cellTypes.csv"
b2_cnames_filename = 'asapseq_cnames.csv'
gnames_filename    = 'gnames.csv'


########################
# read data 

b1_exprs <- readMM(file = paste0(read_dir, b1_exprs_filename))
b2_exprs <- readMM(file = paste0(read_dir, b2_exprs_filename))
b1_adt    = readMM(paste0(read_dir, b1_adt_filename))
b2_adt    = readMM(paste0(read_dir, b2_adt_filename))
b1_meta <- read.table(file = paste0(read_dir, b1_celltype_filename),sep=",",header=T,row.names=1,check.names = F)
b2_meta <- read.table(file = paste0(read_dir, b2_celltype_filename),sep=",",header=T,row.names=1,check.names = F)

b1_cnames = read.table(file=paste0(read_dir, b1_cnames_filename), sep=',', header=T) 
b2_cnames = read.table(file=paste0(read_dir, b2_cnames_filename), sep=',', header=T)
gnames = read.table(file=paste0(read_dir, gnames_filename), header=T, sep=',')

b1_exprs = as.matrix(t(b1_exprs))
b2_exprs = as.matrix(t(b2_exprs))
b1_adt = as.matrix(t(b1_adt))
b2_adt = as.matrix(t(b2_adt))
rownames(b1_exprs) = paste0('gene', 1:dim(b1_exprs)[1])
colnames(b1_exprs) = b1_cnames$X0
rownames(b1_adt) = paste0('prot', 1:dim(b1_adt)[1])
colnames(b1_adt) = b1_cnames$X0

rownames(b2_exprs) = paste0('gene', 1:dim(b2_exprs)[1])
colnames(b2_exprs) = b2_cnames$X0
rownames(b2_adt) = paste0('prot', 1:dim(b2_adt)[1])
colnames(b2_adt) = b2_cnames$X0

label1 = data.frame(type=b1_meta$CellType)
label2 = data.frame(type=b2_meta$CellType)

# lognormalize gene exprs 
rna.pp = CreateSeuratObject(counts=b1_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
rna.pp = NormalizeData(mcasubset.rna.pp, verbose = FALSE)
b1_exprs_normed = rna.pp[['rna']]@data

# lognormalize gene activity 
atac.pp = CreateSeuratObject(counts=b2_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
atac.pp = NormalizeData(atac.pp, verbose = FALSE)
b2_exprs_normed = atac.pp[['rna']]@data

## start 
b1_exprs_concat = rbind(b1_exprs_normed, b1_adt)
rna.obj = CreateSeuratObject(counts=b1_exprs_concat, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
rna.obj@meta.data[['cell_type']] = b1_meta$CellType 
rna.obj = FindVariableFeatures(rna.obj, selection.method = "vst", nfeatures = 2000,
        		verbose = FALSE)
# var_feat = VariableFeatures(object = rna.obj)
# var_feat = unique(c(var_feat, rownames(b1_adt)))
# VariableFeatures(object = rna.obj) = var_feat

rna.obj = ScaleData(rna.obj, verbose=FALSE)
rna.obj = RunPCA(rna.obj, npcs = 30, verbose = FALSE)


# compute lsi for weight.reduction
# mcasubset.atac = FindTopFeatures(mcasubset.atac, min.cutoff = "q0")
# mcasubset.atac = RunTFIDF(mcasubset.atac)
# mcasubset.atac = RunSVD(mcasubset.atac)

b2_exprs_concat = rbind(b2_exprs_normed, b2_adt)
atac.obj = CreateSeuratObject(counts=b2_exprs_concat, project = "atac", assay = "atac",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
atac.obj = ScaleData(atac.obj, verbose=FALSE)
atac.obj = RunPCA(atac.obj, features=VariableFeatures(object = rna.obj), npcs = 30, verbose = FALSE)

# check values, @data
# max(GetAssayData(object = mcasubset.atac))

anchors <- FindTransferAnchors(reference = rna.obj, query = atac.obj,
    features = VariableFeatures(object = rna.obj),
    reduction='cca')

atac.pred <- TransferData(anchorset = anchors, refdata = rna.obj$cell_type,
    weight.reduction = atac.obj[["pca"]], dims = 1:30)

atac.obj <- AddMetaData(atac.obj, metadata = atac.pred)

atac.obj@meta.data[['cell_type']] = b2_meta$CellType

src_cls_set = unique(rna.obj@meta.data[['cell_type']])
shr_mask = atac.obj@meta.data[['cell_type']] %in% src_cls_set
sum(shr_mask)

n_corr = length(which(atac.obj$predicted.id[shr_mask] == atac.obj$cell_type[shr_mask]))

n_corr / sum(shr_mask)

dir.create('/home/yxh/gitrepo/multi-omics/seurat/outputs/')
write.csv(atac.pred, file='/home/yxh/gitrepo/multi-omics/seurat/outputs/CITE-ASAP.csv', quote=F, row.names=T)

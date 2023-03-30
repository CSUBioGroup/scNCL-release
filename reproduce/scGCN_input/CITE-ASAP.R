library(Matrix)
library(glue)
library(Seurat)
setwd('/home/yxh/gitrepo/multi-omics/scGCN/scGCN')
source('/home/yxh/gitrepo/multi-omics/scGCN/scGCN/data_preprocess_utility.R')

##########################################################################################################
##########################################################################################################
# ================================================
# CITE-ASAP dataset
# ================================================
read_dir = glue("/home/yxh/gitrepo/multi-omics/scJoint-main/data/CITE-ASAP/")

b1_exprs_filename = "citeseq_expr.mtx" # not normalized, fuck
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

b1_exprs_concat = rbind(b1_exprs_normed, b1_adt)
b2_exprs_concat = rbind(b2_exprs_normed, b2_adt)

count.list = list(b1_exprs_concat, b2_exprs_concat)
label.list = list(label1, label2)

save_processed_data(count.list, label.list, pp.norm=FALSE, check_unknown=TRUE)

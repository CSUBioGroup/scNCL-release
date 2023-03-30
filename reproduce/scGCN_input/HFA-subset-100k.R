# ================================================
# HumanFetal_100k dataset
# ================================================
library(Matrix)
library(glue)
setwd('/home/yxh/gitrepo/multi-omics/scGCN/scGCN')
source('/home/yxh/gitrepo/multi-omics/scGCN/scGCN/data_preprocess_utility.R')

read_dir = "/home/yxh/data/HumanFetal_100k/"

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

b1_exprs = as.matrix(t(b1_exprs))
b2_exprs = as.matrix(t(b2_exprs))
rownames(b1_exprs) = gnames$X0
colnames(b1_exprs) = b1_cnames$X0
rownames(b2_exprs) = gnames$X0
colnames(b2_exprs) = b2_cnames$X0

label1 = data.frame(type=b1_meta$Main_cluster_name)
label2 = data.frame(type=b2_meta$cell_type)

count.list = list(b1_exprs, b2_exprs)
label.list = list(label1, label2)

save_processed_data(count.list, label.list, check_unknown=FALSE)

# ================================================
# PBMCMultome dataset
# ================================================
library(Matrix)
read_dir = glue("/home/yxh/gitrepo/multi-omics/scJoint-main/data/pbmc_10x/")

b1_exprs_filename = "RNA/matrix.mtx"
b2_exprs_filename = "ATAC_GAM/matrix.mtx"

b1_cnames_filename = 'RNA/barcodes.tsv'
b1_celltype_filename = "metadata.csv"

b2_cnames_filename = 'ATAC_GAM/barcodes.tsv'
b1_gnames_filename = 'RNA/genes.tsv'
b2_gnames_filename = 'ATAC_GAM/genes.tsv'


########################
# read data 

b1_exprs <- readMM(file = paste0(read_dir, b1_exprs_filename))
b2_exprs <- readMM(file = paste0(read_dir, b2_exprs_filename))
b1_meta <- read.table(file = paste0(read_dir, b1_celltype_filename),sep=",",header=T,row.names=1,check.names = F)
# b2_meta <- read.table(file = paste0(read_dir, b2_celltype_filename),sep=",",header=T,row.names=1,check.names = F)

b1_cnames = read.table(file=paste0(read_dir, b1_cnames_filename), sep='\t', header=F) 
b2_cnames = read.table(file=paste0(read_dir, b2_cnames_filename), sep='\t', header=F)

b1_gnames = read.table(file=paste0(read_dir, b1_gnames_filename), header=F)
b2_gnames = read.table(file=paste0(read_dir, b2_gnames_filename), header=F)
head(b1_gnames)

shr_gnames = intersect(b1_gnames$V1, b2_gnames$V1)
rownames(b1_exprs) = b1_gnames$V1
colnames(b1_exprs) = b1_cnames$V1
rownames(b2_exprs) = b2_gnames$V1
colnames(b2_exprs) = b2_cnames$V1

b1_exprs = b1_exprs[shr_gnames, ]
b2_exprs = b2_exprs[shr_gnames, ]

label1 = data.frame(type=b1_meta$seurat_annotations)
label2 = data.frame(type=b1_meta$seurat_annotations)

count.list = list(b1_exprs, b2_exprs)
label.list = list(label1, label2)

# rownames(b1_meta) == b1_cnames$V1  # all True

save_processed_data(count.list, label.list, check_unknown=TRUE)
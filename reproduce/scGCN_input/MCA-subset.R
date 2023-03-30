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

# lognormalize gene activity 
atac.pp = CreateSeuratObject(counts=b2_exprs, project = "rna", assay = "rna",
                                  min.cells = 0,min.features = 0,
                                  names.field = 1)
atac.pp = NormalizeData(atac.pp, verbose = FALSE)
b2_exprs_normed = as.matrix(atac.pp[['rna']]@data) 

count.list = list(b1_exprs, b2_exprs_normed)
label.list = list(label1, label2)

save_processed_data(count.list, label.list, pp.norm=FALSE, check_unknown=FALSE)
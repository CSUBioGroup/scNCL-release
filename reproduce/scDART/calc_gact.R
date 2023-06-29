# rm(list = ls())
# gc()

library(Matrix)
library(BiocGenerics)
library(GenomicRanges)
library(IRanges)


Extend <- function(
    x,
    upstream = 0,
    downstream = 0,
    from.midpoint = FALSE
) {
    if (any(strand(x = x) == "*")) {
        warning("'*' ranges were treated as '+'")
    }
    on_plus <- strand(x = x) == "+" | strand(x = x) == "*"
    if (from.midpoint) {
        midpoints <- start(x = x) + (width(x = x) / 2)
        new_start <- midpoints - ifelse(
            test = on_plus, yes = upstream, no = downstream
        )
        new_end <- midpoints + ifelse(
            test = on_plus, yes = downstream, no = upstream
        )
    } else {
        new_start <- start(x = x) - ifelse(
            test = on_plus, yes = upstream, no = downstream
        )
        new_end <- end(x = x) + ifelse(
            test = on_plus, yes = downstream, no = upstream
        )
    }
    IRanges::ranges(x = x) <- IRanges::IRanges(start = new_start, end = new_end)
    x <- trim(x = x)
    return(x)
}

find_geneact <- function(peak.df, annotation.file, seq.levels, split="-", upstream = 2000, downstream = 0, verbose = FALSE){
    # peak.df is the regions
    peak = peak.df
    # reformualte peak.df of the form "chromosome", "start", "end"
    peak.df <- do.call(what = rbind, args = strsplit(x = peak.df, split = split))
    peak.df <- as.data.frame(x = peak.df)
    colnames(x = peak.df) <- c("chromosome", "start", "end")
    # print(head(peak.df))
    
    # peak.df -> peaks.gr
    peaks.gr <- GenomicRanges::makeGRangesFromDataFrame(df = peak.df)
    BiocGenerics::start(peaks.gr[BiocGenerics::start(peaks.gr) == 0, ]) <- 1
    
    # gtf stores the annotation (reference genome)
    gtf <- rtracklayer::import(con = annotation.file)
    gtf <- GenomeInfoDb::keepSeqlevels(x = gtf, value = seq.levels, pruning.mode = "coarse")
    if (!any(GenomeInfoDb::seqlevelsStyle(x = gtf) == GenomeInfoDb::seqlevelsStyle(x = peaks.gr))) {
        GenomeInfoDb::seqlevelsStyle(gtf) <- GenomeInfoDb::seqlevelsStyle(peaks.gr)
    }
    # gtf.genes stores the genes 
    gtf.genes <- gtf[gtf$type == "gene"]
    
    # update the regions correspond to each gtf.genes, gtf.body_prom
    gtf.body_prom <- Extend(x = gtf.genes, upstream = upstream, downstream = downstream)
    
    # assign peaks.gr to nearest gene region
    gene.distances <- GenomicRanges::distanceToNearest(x = peaks.gr, subject = gtf.body_prom)
    # only leave the ones(regions) overlap with the gene regions(distance = 0)
    keep.overlaps <- gene.distances[rtracklayer::mcols(x = gene.distances)$distance == 
                                        0]
    peak.ids <- peaks.gr[S4Vectors::queryHits(x = keep.overlaps)]
    gene.ids <- gtf.genes[S4Vectors::subjectHits(x = keep.overlaps)]
    gene.ids$gene_name[is.na(gene.ids$gene_name)] <- gene.ids$gene_id[is.na(gene.ids$gene_name)]
    peak.ids$gene.name <- gene.ids$gene_name
    peak.ids <- as.data.frame(x = peak.ids)
    peak.ids$peak <- peak[S4Vectors::queryHits(x = keep.overlaps)]
    # new annotations should include peaks and corresponding gene.name
    annotations <- peak.ids[, c("peak", "gene.name")]
    
    return(annotations)
}


# hyper-parameters
species <- "Mouse"
# upstream region size (base-pair)
upstream <- 2000
# downstream region size (base-pair)
downstream <- 0

# pbmc regions
path0 = './'
path = './pbmc_regions.txt'
regions <- read.table(file = path, sep = ",", header = FALSE)[[1]]
A = find_geneact(
        peak.df = regions, 
        annotation.file = paste0(path0, "Mus_musculus.GRCm38.84.gtf"), 
        seq.levels = c(1:19, "X", "Y"), split='_', upstream = upstream, downstream = downstream, verbose = TRUE
        )
write.table(A, file = paste0(path0, "gact.csv"), sep = ",")

# HFA-subset-50k
path0 = './'
path = './HFA50k_regions.txt'
# regions <- read.table(file = paste0(path, "regions_hvp.txt"), sep = ",", header = FALSE)[[1]]
regions <- read.table(file = path, sep = ",", header = FALSE)[[1]]

A = find_geneact(
    peak.df = regions, 
    annotation.file = paste0(path0, "Homo_sapiens.GRCh37.82.gtf"), 
    seq.levels = c(1:22, "X", "Y"), split='-', upstream = upstream, downstream = downstream, verbose = TRUE
    )

# output gene activity matrix
write.table(A, file = paste0(path0, "gact.csv"), sep = ",")

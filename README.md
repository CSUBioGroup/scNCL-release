# scNCL
A semi-supervised framework for cross-modal label transfer. 

## Installation

Ensure Pytorch is installed in your python environment (our test version: 1.7.1 and 1.12.0). Then installing the basic dependencies:
```
pip install -r requirements.txt
```

We use [`faiss`](https://github.com/facebookresearch/faiss) to accelerate kNN computation. Install instructions: [`INSTALL.md`](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

## Datasets
All datasets used in our paper can be found in [`zenodo`](https://zenodo.org/record/7431624)

Data source:

MCA: [`RNA`](https://tabula-muris.ds.czbiohub.org/) and [`ATAC`](https://atlas.gs.washington.edu/mouse-atac/)

HFA: [`RNA`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156793) and [`ATAC`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149683)

PBMC: [`RNA+ATAC`](https://satijalab.org/seurat/articles/atacseq_integration_vignette.html)

CITE-ASAP: [`RNA+ATAC`](https://github.com/SydneyBioX/scJoint)

## Tutorial
We provide multiple demos to help reproduce our experiments.
* 1. [`CITE-ASAP`](./demo/CITE-ASAP.ipynb)
* 2. [`HFA-subset-50k`](./demo/HFA-subset-50k.ipynb)
* 3. [`MCA`](./demo/MCA.ipynb)
* 4. [`MCAOS`](./demo/MCAOS.ipynb)
* 5. [`MCA-subset`](./demo/MCAsubset.ipynb)
* 6. [`PBMC`](./demo/PBMC.ipynb)

## Usage
Following is instructions about applying scNCL for new datasets:

```Python
# read data
adata_rna    = sc.read_h5ad('data/MCA/scjoint/data_subset/adata_rna.h5ad')   # scrna-seq data, as a example
adata_atac   = sc.read_h5ad('data/MCA/scjoint/data_subset/adata_atac.h5ad')  # scatac-seq data

# regarding the basic pipeline for computing low-dimensional representations of scATAC-seq raw data, 
# please refer to Seurat's tutorial (https://satijalab.org/seurat/articles/atacseq_integration_vignette.html)
atac_raw_emb = np.load('data/MCA/scjoint/data_subset/atac_raw_emb.npy')     # pca matrix or tSNE coordinates

model = scNCL.scNCL(
                'non_linear', n_latent=64, bn=False, dr=0.2, 
                cont_w=0.05, cont_tau=0.4,
        )
    
# dictionary of preprocessing parameter
ppd = {'binz': True, 
       'hvg_num':adata_atac.shape[1], 
       'lognorm':False, 
       'scale_per_batch':False,  
       'batch_label': 'domain',
       'type_label':  'cell_type',
       'knn': 10,
       'knn_by_tissue':False
}  # default settings

model.preprocess(
                [adata_rna, adata_atac], 
                atac_raw_emb,   
                adata_adt_inputs=None,   # protein input, e.g., [adata_protein_ref, adata_protein_tgt]
                pp_dict = ppd          
)

# model training
model.train(
        opt='adam', 
        batch_size=500, training_steps=1000, 
        lr=0.001, weight_decay=5e-4,
        log_step=50, eval_atac=False, 
)

# model inference and computing inferred cell labels. 
model.eval(inplace=True)
atac_pred_type = model.annotate()
```

## Important Arguments
1. Arguments for scNCL object:
* `encoder_type`:  architecture choice of encoder network. optional={'linear', 'non-linear'}
* `cont_w`:        weight for contrastive learning loss term 
* `cont_tau`:      temperature parameter for contrastive learning
* `align_w`:       weight for feature alignment loss
* `align_p`:       fractions of scATAC-seq cells for further alignment with scRNA-seq cells

2. Arguments for preprocessing
* `binz`:          binarizing raw gene expression matrix or gene activity matrix with threshold 0
* `knn_by_tissue`: whether computing kNN graph within each tissue

3. Arguments for training
* `adam`:          optimizer, 'adam' for non-linear encoder, and 'sgd' for linear encoder
* `training_steps`:number of training steps
* `lr`:            learning rate
* `eval_atac`:     whether evaluate the prediction results during training. 


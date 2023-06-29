# scNCL
A semi-supervised framework for cross-modal label transfer. 

## Installation
1. clone repository

`git clone https://github.com/CSUBioGroup/scNCL-release.git
cd scNCL-release/`

2. create env

`conda create -n scNCL python=3.7.3
source activate scNCL`

3. install pytorch (our test-version: torch==1.7.1+cu101, cuda: 10.1)

`pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

4. install other dependencies

`pip install -r requirements.txt`

5. setup

`python setup.py install`

* Note that if reporting `ERROR: Could not find a version that satisfies the requirement nose>=1.0`, please run command:
`conda install nose`

## Datasets
All datasets used in our paper can be found in [`zenodo`](https://zenodo.org/record/7787402)

Data source:

MCA: [`RNA`](https://tabula-muris.ds.czbiohub.org/) and [`ATAC`](https://atlas.gs.washington.edu/mouse-atac/)

HFA: [`RNA`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156793) and [`ATAC`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149683)

PBMC: [`RNA+ATAC`](https://satijalab.org/seurat/articles/atacseq_integration_vignette.html)

CITE-ASAP: [`RNA+ATAC`](https://github.com/SydneyBioX/scJoint)

## Demo
We provide a demo to help users test our code: [`PBMC`](./Examples/PBMC-demo.ipynb). Data used in the demo is already provided in the [`demo_data`](./Examples/demo_data) folder. 

## Tutorial
We provide examples to help reproduce our experiments.
* 1. [`CITE-ASAP`](./Examples/CITE-ASAP.ipynb)
* 2. [`HFA-subset-50k`](./Examples/HFA-subset-50k.ipynb)
* 3. [`MCA`](./Examples/MCA.ipynb)
* 4. [`MCAOS`](./Examples/MCAOS.ipynb)
* 5. [`MCA-subset`](./Examples/MCAsubset.ipynb)

## Usage
Following is instructions about applying scNCL for new datasets:

```Python
# read data
adata_rna    = sc.read_h5ad('data/MCA/scjoint/data_subset/adata_rna.h5ad')   # scrna-seq data, as a example
adata_atac   = sc.read_h5ad('data/MCA/scjoint/data_subset/adata_atac.h5ad')  # scatac-seq data

# regarding the basic pipeline for computing low-dimensional representations of scATAC-seq raw data, 
# please refer to Seurat's tutorial (https://satijalab.org/seurat/articles/atacseq_integration_vignette.html)
atac_raw_emb = np.load('data/MCA/scjoint/data_subset/atac_raw_emb.npy')     # pca matrix or tSNE coordinates

model = BuildscNCL(
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


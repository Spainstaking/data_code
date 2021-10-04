paper "A network-based deep learning methodology for stratification of tumor mutations"

'dataset' directory
Contain the somatic mutation data、human protein–protein interactome network data and 
patient clinical data.

'struc2vec' directory
Contain the protein–protein interactome network embedding method.Each gene generates its 
own characteristics.
python struc2vec/src/main.py --input struc2vec/graph/Human_Interactome.txt --output struc2vec/emb/gene_emb.txt 
--num-walks 20 --walk-length 80 --window-size 5 --dimensions 2 --OPT1 True --OPT2 True --OPT3 True --until-layer 6

'patient_feature' directory
Contain the patient feature construction method.
python patient_feature/patient_feature.py

Tutorial
1.To get gene specific expression learned by gene_specific.py.
2.Tumor classification across various cancer types learned by tumour_classification.py.
3.Tumor classification for the specific cancer type learned by specific_patient_classificationA.py.
4.Survival differences between patients learned by patient_cluster.py.

Requirements
This work is tested to work under Python 3.7
The required dependencies for deepDR are gensim, lifelines, pandas, numpy, scipy, and scikit-learn.
EEG-to-fMRI Deep Learning Model for Ventral Striatum Reward Prediction

Code & Data Repository for “From surface to depth: using deep learning to predict striatal fMRI reward signaling from EEG” (Herzog et al., in prep)

The project adapts BEIRA, a convolutional autoencoder model (Kovalev et al., 2022), to reconstruct ventral striatum (VS) BOLD activity from simultaneous EEG during a well-validated two-choice gambling reward task. 
Unlike subject-specific models, this work trains a single group-level model and evaluates its cross-participant generalization.

Data files were too big to be uploaded directly here.
Only weight files are uploaded here.

You can find the preprocessed and shaped data here:
https://osf.io/8sn9h/overview



/
├── code/               
│   ├── main_train.py   
│   ├── utils/
|       ├── train_utils.py
|       ├── torch_dataset.py
|       ├── inference.py
|       ├── preproc.py
|       ├── model_arch/
|           ├── autoencoder_new_ArturNH.py 
|
├── data/
│   ├── DL_model/ 
|       ├── best_perm/
|           ├── weights_for_best_perm.pt 
|       ├── worst_perm/
|           ├── weights_for_worst_perm.pt
|       ├── rand_perm/
|           ├── weights_for_random_perm.pt
│   └── linear_model/        
│
└── README.md

This repository contains the official codebase and model weights for the study:

**From surface to depth: using deep learning to predict striatal fMRI reward signaling from EEG.**

*Herzog et al., in prep.*

The project adapts BEIRA, a convolutional autoencoder architecture originally proposed by Kovalev et al. (2022, https://github.com/kovalalvi/beira?tab=readme-ov-file), to reconstruct ventral striatum (VS) BOLD activity from simultaneously acquired EEG data recorded during a well-validated two-choice gambling reward task. 

✔ Included in this repository:
- Training and evaluation scripts
- Model architecture
- final trained model weights for the best, a random, and the worst permutation folds



✔ Available externally:

Preprocessed datasets were too large to upload to GitHub.
All preprocessed EEG, fMRI (VS time series), and model-ready data files are hosted on OSF:

🔗 https://osf.io/8sn9h/overview

For starting training procees you need to 

    1) Clone repository and create virtual environment
    
    2) Download data from OSF and move to data folder in your repository.
    
    3) edit base_directory in code/main_train.py
    
    4) create wandb key and insert key in code/main_train.py
    
    5) run main_train.py

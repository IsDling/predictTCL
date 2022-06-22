# predictTCL
Implementation for Joint Prediction of Meningioma Grade and Brain Invasion via Task-Aware Contrastive Learning in pytorch. MICCAI 2022 accept.

**Requirements**
- Python 3
- PyTorch
- SimpleITK
- scikit-learn
- scipy

**File structure**

-sup

-----contrastive_loss.py

-----data_provider.py

-----make_randomfolders.py

-----online_aug.py

-----predictTCL.py

-----resnet3d.py

-----scheduler.py

-main.py

**Running the Code**

-Firstly, run make_randomfolders.py to make 3 random folder data. 

-Then run python main.py to train and test. Change the parameter ‘run_type’ in main.py to ‘train’ for train and ‘test’ for test.

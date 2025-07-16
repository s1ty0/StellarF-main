Stellar-F Paper Replication Process 
Before we start our test, please make sure that the conditions for data preprocessing are met. 
Please execute:

```
conda create -n my_base python=3.10
conda activate my_base
pip install -r requirements.txt
```



1. Obtaining the original optical variation data. 
   The original Kepler and TESS light curve data can be obtained from the anonymous Google Drive at this link: https://drive.google.com/drive/folders/1rDFYRoEruB8_pVxlfOcS5OnsHleIhITz?usp=sharing 
   After obtaining the original data, please place it in the directories `./LLM/raw_data/` and `./TS-Lib/raw_data/` and unzip it. Similarly, create several new folders respectively to store the data we are about to obtain, as shown in the following tree diagram. The directory structure after unzipping should be as follows:

   ```
   LLM
   |─raw_data                                                     
   |  ├─kepler                                                    
   |  └─tess
   |─dataset_k   
   |  ├─train                                                    
   |  └─val
   |  └─test
   |─dataset_t
   |  ├─train                                                    
   |  └─val
   |  └─test
   |─dataset_k12  
   |  ├─train                                                    
   |  └─val
   |  └─test
   |─dataset_t12   
   |  ├─train                                                    
   |  └─val
   |  └─test
   
   TS-Lib
   |─raw_data                                                     
   |  ├─kepler                                                    
   |  └─tess
   |─dataset_k   
   |  ├─train                                                    
   |  └─val
   |  └─test
   |─dataset_t   
   |  ├─train                                                    
   |  └─val
   |  └─test
   |─dataset_k12   
   |  ├─train                                                    
   |  └─val
   |  └─test
   |─dataset_t12   
   |  ├─train                                                    
   |  └─val
   |  └─test
   ```
   
   
   
2. Data Preprocessing 
    The data preprocessing we adopted includes filling in missing values and patch division. 
    For the raw data in `./LLM/raw_data/` and `./TS-Lib/raw_data/`, please execute the following commands respectively to obtain the Kepler and TESS data after linear interpolation and patch division. They are stored in `dataset_k` and `dataset_t` respectively. 

  ```
  cd ./LLM # Switch to the LLM and TS-Lib folders under the home directory.
  python data_process.py
  ```

  ```
  cd ./TS-Lib
  python data_process.py
  ```

  Once the data has been linearly interpolated and divided into patches, it is possible to add new points! 
  Please execute the following commands to obtain the Kepler and TESS data after adding the statistics information of Innovation Point 1 - Flare and the historical records of Innovation Point 2 - Flare. Name them as `dataset_k12` and `dataset_t12` respectively.

  ```
  cd ./LLM # Switch to the LLM and TS-Lib folders under the home directory too.
  python data_process_12.py
  ```

  ```
  cd ./TS-Lib
  python data_process_12.py
  ```



3. Model Testing and Evaluation 
    I'm sure you have successfully completed the construction of our data! 
    Next, if you want to replicate the models in TS-Lib, please go to `./TS-Lib/README.md` 
    If you want to replicate the large language model, please go to `./LLM/README.md` instead.


Before we start our experiment, please activate our conda.

```
conda activate my_base
```

Please reproduce according to the following instructions. 
First of all, please download the weights of the corresponding large model to your local machine. (from hugging face)
And construct the following file tree as the folder for storing the pre-trained weights.

```
├─bert_base_uncased
├─deberta_v3_base
├─gpt2_local
├─moment_base
├─RoBERTa_base
```

Please follow the following instructions to reproduce.
Then execute the code separately to reproduce the moment and gpt4ts.

```
python moment_tu.py
python gpt4ts_tu.py
```

and

```
python moment_12.py
python gpt4ts_12.py
```

Just switch the corresponding dataset parameters, and this has been indicated in the corresponding code. 
Execute the following codes separately to reproduce the fine-tuning of the large model.

```
bert_tu12_l.py
deberta_tu12_l.py
gpt2_tu12_l.py
roberta_tu12_l.py
```

OK, that's it. Basically, you can reproduce all the main experimental results! If you have any questions, feel free to ask!!
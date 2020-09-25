
# Language Variety Identification 

This project was conducted to study the use of the AllenNLP (PyTorch) tool for a classification task. It consists of learning to discriminate between text samples writen in Brazilian Portuguese or European Portuguese (Portugal).


# Dataset
You should download the DSL Shared Task dataset(http://ttg.uni-saarland.de/resources/DSLCC/) (I have used the version DSLCC v2.1 - the 2016 edition) and adjust the config.json file with the paths for the training, validation and testing (gold standard) text files. You can check the config.json file for an example.

## Data Preprocessing
After download the dataset, we filter the portuguese samples using the `utils/preprocessing.py` sript, which receives as argument the basepath containing the DSLCC traind and gold folders. 
Example:
```
python utils/preprocessing.py --data_dir DSL-Task/data/DSLCC-v2.1
```
# Training
The training details are specified in the config.json, as well as the serialization directory where we persist the model checkpoints, vocabulary and logging details. We can perform the training with the following command:
```
python train.py --config config.json --serialization_dir weights
```

# Standalone Validation
If you already have a trained model, you can directly assess its performance against the validation partition using the following script:

```
python eval.py --config config.json --serialization_dir weights --model_checkpoint best.th
```

# Command-line demo (unlabeled data)
Finally, you can also sanity check the model by providing a text sample and getting the classification results as output.
Example;
```
python demo.py --serialization_dir weights --model_checkpoint best.th --text "Oi, tudo bem com você?"
```
output
![demo output](https://github.com/wellescastro/language_variety_classification/raw/master/demo-example-output.png)

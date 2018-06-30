'''Author: Mohamed Salem Elaraby'''
'''email: mohamed.elaraby@alumni.ubc.ca'''
'''Date: 27th June 2018'''

## Example of using Text Classification library of the shelf

set -e

model=attbilsm
EmbeddingModelDir=./EmbeddingModels


python3.4 test_baselines.py --train data/train/MultiTrain.Shuffled.csv --Ar='True' --dev data/dev/MultiDev.csv --test data/test/MultiTest.csv --model_type=$model --static=False --rand=False --embedding=$EmbeddingModelDir/CbowModel.mdl --model_file=$model'_CBOWModel_non_static_Multi_Model'


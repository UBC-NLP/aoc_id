'''Author: Mohamed Salem Elaraby'''
'''email: mohamed.elaraby@alumni.ubc.ca'''
'''Date: 27th June 2018'''

## Example of using Text Classification library of the shelf

set -e

model=blstm
EmbType=fastText
EmbeddingModelDir=/EmbeddingModels/
DataDir=data/kickstarter

python3.4 test_baselines.py --train $DataDir/train_data.csv --Ar='False' --dev $DataDir/valid_data.csv --test $DataDir/test_data.csv --model_type=$model --static=False --rand=False --EMB_type=$EmbType --embedding=$EmbeddingModelDir/crawl-300d-2M.vec --model_file=$model'_kickstarter_fasttext_init'


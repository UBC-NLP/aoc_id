# Text-Classification-of-the-shelf-with-a-normalizer-for-Arabic-text-

*This is a simple text classification library based on keras associated with a text normalization script for Arabic*

# Current Implemented Models:

1- Word Level CNN based on:
"Convultion Neural Network for Text Classificartion"
url: http://www.aclweb.org/anthology/D14-1181

2- Word Level C-LSTM based on:
"A C-LSTM Neural Network for Text Classification"
url:https://arxiv.org/pdf/1511.08630.pdf

3- Recurrent Network and its variants (BiLSTM, LSTM, GRU, BiGRU, Attention-BiLSTM)

4- Models implemented but currently not supported in options (Attention-LSTM,Attention-BiGRU).

5- Not yet tested  (char level CNN). 

# Requirements

- keras (2.0 or above)
- gensim
- numpy
- pandas

# General Usage:
- * Tested with python 3.4 *
- python test_baselines.py --train training_file --Ar='True' --dev Dev_File --test test_file --model_type=model_selection --static=Trainable_embeddings --rand=Random_Embeddings --embedding=External_Embedding_model --model_file=Output_model_file_inJson
- put your training labels in [link:][https://github.com/EngSalem/Text-Classification-of-the-shelf-with-a-normalizer-for-Arabic-text-/blob/master/conf/label_list]
# Options details #

- train: training file assuming in csv format, text, label
- Ar: if True then Arabic normalization is applied (should be true in case of external embeddings)
- dev: Development file in csv format 
- test: test file in csv format
- model_type: currently support those type of models: (cnn: word level cnn, clstm: word level clstm, lstm: vanilla lstm architecture, blstm: Vanilla bidirectional LSTM, bigru: Vanilla BiGated Recurrent unit, attbilstm: BiLSTM with self attention mechanism)
- static: used in case of external embedding, if True: External Embeddings are not fine tuned during training, if False: External EMbeddings are fine tuned during training). 
- rand: if True, No external embedding is applied, randomly initialized embedding 
- embedding: External embedding model in gensim format
- model_file: Output model file in Json.

*Note: final model score is dumped into a file with name_of_model_score with both dev and test scores*
# Example Project (Arabic Dialect Identification with Deep Models) #

- This project utilize 6 deep learning models applied on Arabic Online Commentary Dataset 
- url:  https://www.cs.jhu.edu/~ccb/publications/arabic-dialect-corpus.pdf
- dataset url: https://www.cis.upenn.edu/~ccb/data/AOC-dialectal-annotations.zip 
- make sure to cite AOC oringial paper if you are going to use it in your work. 
- This work currently accepted to VarDial Worshop 2018 co-located with COLING 2018 under the name (paper soon)
"Deep Models for Arabic Dialect Identification on Benchmarked Data"
- Training data link: [link]: https://github.com/EngSalem/Text-Classification-of-the-shelf-with-a-normalizer-for-Arabic-text-/tree/master/data/train
- Dev data link: [link]: https://github.com/EngSalem/Text-Classification-of-the-shelf-with-a-normalizer-for-Arabic-text-/tree/master/data/dev
- Test data link: [link]: https://github.com/EngSalem/Text-Classification-of-the-shelf-with-a-normalizer-for-Arabic-text-/tree/master/data/test
- An example on how to use it is in:  https://github.com/EngSalem/Text-Classification-of-the-shelf-with-a-normalizer-for-Arabic-text-/blob/master/run.sh

- *If you are going to follow up on this project please cite this work using the following bibtext:*

@inproceedings{Elaraby2018,
  title={Deep Models for Arabic Dialect Identification on Benchmarked Data},
  author={Elaraby, Mohamed and Abdul-Mageed, Muhammad},
  booktitle={Proceedings of the Fifth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial5)},
  year={2018}
}

# External Embedding Models #

- For Arabic Dialects we release 2 embedding models 
- AOC embedding: Download in url: [link] https://drive.google.com/open?id=1QEg9HotnTCI45-PT52g445bp5qYQ4RSm
- Twitter Embedding Model: Download in url: [link]  https://drive.google.com/open?id=1hEuNHn2PA7kIf1IK0FUGUskA77YZJ3vO
 - cite the following paper if you are planning to use city level dialect embedding model: 
 
@article{abdulyou,
  title={You Tweet What You Speak: A City-Level Dataset of Arabic Dialects},
  author={Abdul-Mageed, Muhammad and Alhuzali, Hassan and Elaraby, Mohamed}
}

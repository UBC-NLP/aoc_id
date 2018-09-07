import data_helpers as dh
from BaselineModels import cnn_kim,clstm,BasicLSTM,BasicBiLSTM, BasicBiGRUs, AttentionBiLSTM
import numpy as np
import argparse
################################
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store', dest='train_corpus',
                    help='Input train file', default='data/train/TrainBinary.Shuffled.csv')
parser.add_argument('--Ar', action='store', dest='Ar', help='Choice if apply arabic normalziation', default='False')
parser.add_argument('--dev',action='store',dest='dev_corpus',
                     help='Input dev file',default='data/dev/DevBinary.csv')
parser.add_argument('--test',action='store',dest='test_corpus',
                     help='Input test file',default='data/test/TestBinary.csv')
parser.add_argument('--model_type',action='store',dest='model_type',
                     help='Baseline Model type', default='cnn')
parser.add_argument('--static',action='store',dest='STATIC',
                    help='STATIC embedding or non static for external embedding',default='True')

parser.add_argument('--rand',action='store',dest='rand',
                    help='Random Initialization for embedding or not')

parser.add_argument('--EMB_type',action='store',
                    help='embedding type, choice between skipgram CBOW or fastText' , default='CBOW')

parser.add_argument('--embedding',action='store',dest='embedd_file',
                    help='Embedding Model',default="AOC_Skipgram.mdl")


parser.add_argument('--model_file',action='store',dest='ModelFile',
                    help='The output of the model file',default='models/CNN_Model')


args = parser.parse_args()
#====================================#
Arabic=False
if args.Ar=='True':
    Arabic=True
else:
    Arabic=False
#====================================#
print('----- Load Train and Test Data --------')
X_train,Y_train,Y_train_true=dh.LoadData(args.train_corpus,ClassesDict=dh.get_classes(),Arabic=Arabic)
X_valid,Y_valid,Y_valid_true=dh.LoadData(args.dev_corpus,ClassesDict=dh.get_classes(),Arabic=Arabic)
X_test, Y_test, Y_test_true = dh.LoadData(args.test_corpus, ClassesDict=dh.get_classes(),Arabic=Arabic)

#print(X_valid)
print('---- Tokenizing Training and Testing Data ------')
X_train, X_valid,X_test,wordmap = dh.tokenizeData(X_train, X_valid, vocab_size=dh.get_vocab_size(),X_test=X_test)
X_train, X_valid,X_test = dh.paddingSequence(X_train, X_valid, maxLen=30,X_test=X_test)
n_symbols,word_map=dh.get_word_map_num_symbols(args.train_corpus)
###############################
RAND=True
###########################
if args.rand=='True':
   RAND=True 
elif args.rand=='False':
   RAND=False
##########################
Trainable=False
if args.STATIC=="True":
   Trainable=False
elif args.STATIC=="False":
   Trainable=True
##########################


if args.model_type=="cnn":
   FW=open("CNN_scores",'w')
   CNNBaseline=cnn_kim(cnn_rand=RAND,STATIC=Trainable,ExternalEmbeddingModel=args.embedd_file,EmbeddingType=args.EMB_type,n_symbols=n_symbols,wordmap=word_map)
   CNNBaseline.train_model(CNNBaseline.model,X_train,Y_train=Y_train,X_valid=X_valid,Y_valid=Y_valid)
   ValidScore=CNNBaseline.Evaluate_model(CNNBaseline.model,X_valid,Y_valid)
   TestScore=CNNBaseline.Evaluate_model(CNNBaseline.model,X_test,Y_test)
   FW.write('CNN Validation score: '+str(ValidScore)+"\n")
   FW.write('CNN Test score: '+str(TestScore)+"\n")
   FW.close()
   CNNBaseline.save_model(args.ModelFile,CNNBaseline.model)
elif args.model_type=="clstm":
   FW=open("CLSTM_scores","w")
   CLSTMBaseline=clstm(clstm_rand=RAND,STATIC=Trainable,ExternalEmbeddingModel=args.embedd_file,EmbeddingType=args.EMB_type,n_symbols=n_symbols,wordmap=word_map)
   CLSTMBaseline.train_model(CLSTMBaseline.model,X_train=X_train,Y_train=Y_train,X_valid=X_valid,Y_valid=Y_valid)
   ValidScore=CLSTMBaseline.Evaluate_model(CLSTMBaseline.model,X_valid,Y_valid)
   TestScore=CLSTMBaseline.Evaluate_model(CLSTMBaseline.model,X_test,Y_test)
   FW.write('CLSTM Validation score: ' + str(ValidScore) + "\n")
   FW.write('CLSTM Test score: ' + str(TestScore) + "\n")
   FW.close()
   CLSTMBaseline.save_model(args.ModelFile, CLSTMBaseline.model)
elif args.model_type=="lstm":
   FW=open("LSTM_score",'w')
   LSTMBaseline=BasicLSTM(lstm_rand=RAND,STATIC=Trainable,ExternalEmbeddingModel=args.embedd_file,EmbeddingType=args.EMB_type,n_symbols=n_symbols,wordmap=word_map)
   LSTMBaseline.train_model(LSTMBaseline.model,X_train=X_train,Y_train=Y_train,X_valid=X_valid,Y_valid=Y_valid)
   ValidScore = LSTMBaseline.Evaluate_model(LSTMBaseline.model, X_valid, Y_valid)
   TestScore = LSTMBaseline.Evaluate_model(LSTMBaseline.model, X_test, Y_test)
   FW.write('LSTM Validation score: ' + str(ValidScore) + "\n")
   FW.write('LSTM Test score: ' + str(TestScore) + "\n")
   FW.close()
   LSTMBaseline.save_model(args.ModelFile, LSTMBaseline.model)
elif args.model_type=="blstm":
    FW = open("BLSTM_score", 'w')
    BiLSTMBaseline = BasicBiLSTM(bilstm_rand=RAND, STATIC=Trainable, ExternalEmbeddingModel=args.embedd_file,EmbeddingType=args.EMB_type,n_symbols=n_symbols,wordmap=word_map)
    BiLSTMBaseline.train_model(BiLSTMBaseline.model, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
    ValidScore = BiLSTMBaseline.Evaluate_model(BiLSTMBaseline.model, X_valid, Y_valid)
    TestScore = BiLSTMBaseline.Evaluate_model(BiLSTMBaseline.model, X_test, Y_test)
    FW.write('BiLSTM Validation score: ' + str(ValidScore) + "\n")
    FW.write('BiLSTM Test score: ' + str(TestScore) + "\n")
    FW.close()
    BiLSTMBaseline.save_model(args.ModelFile, BiLSTMBaseline.model)
elif args.model_type=="bigru":
    FW = open("BiGRU_score", 'w')
    BiGRUBaseline = BasicBiGRUs(BiGRU_rand=RAND, STATIC=Trainable, ExternalEmbeddingModel=args.embedd_file,EmbeddingType=args.EMB_type,n_symbols=n_symbols,wordmap=word_map)
    BiGRUBaseline.train_model(BiGRUBaseline.model, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
    ValidScore = BiGRUBaseline.Evaluate_model(BiGRUBaseline.model, X_valid, Y_valid)
    TestScore = BiGRUBaseline.Evaluate_model(BiGRUBaseline.model, X_test, Y_test)
    FW.write('BiGRU Validation score: ' + str(ValidScore) + "\n")
    FW.write('BiGRU Test score: ' + str(TestScore) + "\n")
    FW.close()
    BiGRUBaseline.save_model(args.ModelFile, BiGRUBaseline.model)
elif args.model_type=="attbilsm":
    FW = open("attbilstm_score", 'w')
    AttBiLSTM = AttentionBiLSTM(att_rand=RAND, STATIC=Trainable, ExternalEmbeddingModel=args.embedd_file,EmbeddingType=args.EMB_type,
                                n_symbols=n_symbols, wordmap=word_map)
    AttBiLSTM.train_model(AttBiLSTM.model, X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
    ValidScore = AttBiLSTM.Evaluate_model(AttBiLSTM.model, X_valid, Y_valid)
    TestScore = AttBiLSTM.Evaluate_model(AttBiLSTM.model, X_test, Y_test)
    FW.write('BiGRU Validation score: ' + str(ValidScore) + "\n")
    FW.write('BiGRU Test score: ' + str(TestScore) + "\n")
    FW.close()
    AttBiLSTM.save_model(args.ModelFile, AttBiLSTM.model)

##CNN

#FirstBaseline=cnn_kim(cnn_rand=False,STATIC=True)
#FirstBaseline.train_model(FirstBaseline.model,X_train=X_train,Y_train=Y_train,X_valid=X_test,Y_valid=Y_test)

##C-LSTM

#SecondBaseline=clstm(clstm_rand=False,STATIC=True)
#SecondBaseline.train_model(SecondBaseline.model,X_train=X_train,Y_train=Y_train,X_valid=X_test,Y_valid=Y_test)

##BasicLSTM

#ThirdBaseline=BasicLSTM(lstm_rand=False,STATIC=True)
#ThirdBaseline.train_model(ThirdBaseline.model,X_train=X_train,Y_train=Y_train,X_valid=X_test,Y_valid=Y_test)

##BasicBiLSTM

#FourthBaseline=BasicBiLSTM(bilstm_rand=False,STATIC=False)
#FourthBaseline.train_model(FourthBaseline.model,X_train=X_train,Y_train=Y_train,X_valid=X_test,Y_valid=Y_test)

##

#FifthBaseline=BasicBiGRUs(BiGRU_rand=False,STATIC=False)

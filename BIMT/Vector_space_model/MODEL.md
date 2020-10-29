## Query

On query we just removed stopwords and sent tokenized words within the csv file. On expected docs of each query we put weight 2 for all evaluation except for colleagues of REW that we put 1. Also we normalized all evaluation scores to be from [0,1] and work like a probability.

![](https://latex.codecogs.com/gif.latex?score_i%20%3D%20%5Cfrac%7Bscore_i%7D%7B%5Csum_%7B%5Cforall%20j%7D%20score_j%7D)

## Iverse Index Generator

Here we also only removed stopwords from our corpus and we check for word frequency on each document.
We created 2 objects for this task: _inverted_index_ and _doc_index_. Inverted index has a word and a dict with documents it appears with its frequency in the document. On doc_index it has a document number and a dict with the words that are in this document and that word frequency on it.

## Model

We used both objects created previously saved on _doc_index.csv_ and _inverted_index.csv_. Here we created TF-IDF representation of the words using inverted_index. Then we used our doc_index object to transform each word frequency on each word TF-IDF value in each document. We also used the third recommended tf-idf wheight scheme that can be seen above:


![](https://latex.codecogs.com/gif.latex?w_%7Bt%2Cd%7D%20%3D%20%281%20&plus;%20%5Clog_2%20f_%7Bt%2Cd%7D%29%5Ccdot%20%5Clog%20%5Cfrac%7BN%7D%7Bn_t%7D)


## Search Engine

This one is pretty straight since we have all files compiled, we just _use vectorial_model_ object and compute de query similarity with the documents, we utilize the cossine distance for this and only retrieve the top 20 documents on rank.
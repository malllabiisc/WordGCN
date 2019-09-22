


<div align="center"> 
  
## Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks  

[![Paper](http://img.shields.io/badge/paper-arxiv.1809.04283-B31B1B.svg)](https://arxiv.org/abs/1809.04283)
[![Conference](http://img.shields.io/badge/ACL-2019-4b44ce.svg)](https://www.aclweb.org/anthology/P19-1320/)
</div>

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/WordGCN/blob/master/images/syngcn_model.png" alt="...">
</p>

*Overview of SynGCN: SynGCN employs Graph Convolution Network for utilizing dependency context for learning word embeddings. For each word in vocabulary, the model learns its representation by aiming to predict each word based on its dependency context encoded using GCNs. Please refer Section 5 of the paper for more details.*

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.
- Install [word-embedding-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks) used for evaluating learned embeddings.
  - The test and valid dataset splits used in the paper can be downloaded from [this link](https://drive.google.com/open?id=1VMyddIOgmkskAFN2BvI6c49Y63SHjNfF). Replace the original `~/web_data` folder with the provided one.  
  - For switching between valid and test split execute `python switch_evaluation_data.py -split <valid/valid>`

### Dataset:

* We used [Wikipedia corpus](https://dumps.wikimedia.org/enwiki/20180301/). The processed version can be downloaded from [here](https://drive.google.com/file/d/1S1UYXc3PfoNFcNY6tB5ahiugXh5qidz-/view?usp=sharing).

* The processed dataset includes:
  * `voc2id.txt` mapping of words to to their unique identifiers.
  * `word2freq.txt` contains frequency of words in the corpus.
  * `de2id.txt` mapping of dependency relations to their unique identifiers. 
  * `data.txt` contains the entire Wikipedia corpus with each sentence of corpus stored in the following format:

    ```java
    <num_words> <num_dep_rels> tok1 tok2 tok3 ... tokn dep_e1 dep_e2 .... dep_em
    ```

    - Here, `num_words` is the number of words and `num_dep_rels`  denotes the number of dependency relations in the sentence.
    - `tok_1, tok_2 ...` is the list of tokens in the sentence and `dep_e1, dep_e2 ...`is the list of dependency relations where each is of form `source_token|destination_token|dep_rel_label`.

### Training SynGCN embeddings:
- Download the processed Wikipedia corpus ([link](https://drive.google.com/file/d/1S1UYXc3PfoNFcNY6tB5ahiugXh5qidz-/view?usp=sharing)) and extract it in `./data` directory.
- Execute `make` to compile the C++ code for creating batches.
- To start training run:
  ```shell
  python syngcn.py -name test_embeddings -dump -gpu 0
  ```
  
* The trained embeddings will be stored in `./embeddings` directory with the provided name `test_embeddings` .
* **Note:** As reported in TensorFlow issue [#13048](https://github.com/tensorflow/tensorflow/issues/13048). The current SynGCN's TF-based implementation is slow compared to [Mikolov's word2vec](https://github.com/tmikolov/word2vec) implementation. For training SynGCN on a very large corpus might require multi-GPU or C++ based implementation.

### Fine-tuning embedding using SemGCN:

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/WordGCN/blob/master/images/semgcn_model.png" alt="...">
</p>

- Pre-trained 300-dimensional `SynGCN` embeddings can be downloaded from [here](https://drive.google.com/file/d/1wYgdyjIBC6nIC-bX29kByA0GwnUSR9Hh/view?usp=sharing). 
- For incorporating semantic information in given embeddings run:
  ```shell
  python semgcn.py -embed ./embeddings/pretrained_embed.txt 
                   -semantic synonyms -embed_dim 300 
                   -name fine_tuned_embeddings -dump -gpu 0
  ```
* The fine-tuned embeddings will be saved in `./embeddings` directory with name `fine_tuned_embeddings`. 

### Extrinsic Evaluation:

For extrinsic evaluation of embeddings the models from the following papers were used:

* NCR (Neural Co-reference Resolution): [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://github.com/kentonl/e2e-coref).
* NER (Named Entity Recognition): [NeuroNER: an easy-to-use program for named-entity recognition based on neural networks](https://github.com/Franck-Dernoncourt/NeuroNER).
* POS (Parts of speech tagging): [BiLSTM-CNN-CRF architecture for sequence tagging](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf).
* SQuAD (Question Answering): [Simple and Effective Multi-Paragraph Reading Comprehension](https://github.com/allenai/document-qa/tree/master/docqa/elmo)

### Citation:
Please cite the following paper if you use this code in your work.

```
@inproceedings{vashishth-etal-2019-incorporating,
    title = "Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks",
    author = "Vashishth, Shikhar  and
      Bhandari, Manik  and
      Yadav, Prateek  and
      Rai, Piyush  and
      Bhattacharyya, Chiranjib  and
      Talukdar, Partha",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1320",
    pages = "3308--3318"
}
```
For any clarification, comments, or suggestions please create an issue or contact [shikhar@iisc.ac.in](http://shikhar-vashishth.github.io).

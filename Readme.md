## Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks

Source code for [ACL 2019](http://acl2019.org) paper: [Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks](https://arxiv.org/abs/1809.04283).


<p align="center">
  <img align="center" src="https://github.com/malllabiisc/WordGCN/blob/master/images/syngcn_model.png" alt="...">
</p>

*Overview of SynGCN: SynGCN employs Graph Convolution Network for utilizing dependency context for learning word embeddings. For each word in vocabulary, the model learns its representation by aiming to predict each word based on its dependency context encoded using GCNs. Please refer Section 5 of the paper for more details.*

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.
- Install [word-embedding-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks) used for evaluating learned embeddings.

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
  python syngcn.py -name test_embeddings -gpu 0
  ```
  
* The trained embeddings will be stored in `./embeddings` directory with name `test_embeddings` .

### Fine-tuning embedding using SemGCN:

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/WordGCN/blob/master/images/semgcn_model.png" alt="...">
</p>

- Pre-trained 300-dimensional `SynGCN` embeddings can be downloaded from [here](https://drive.google.com/file/d/1wYgdyjIBC6nIC-bX29kByA0GwnUSR9Hh/view?usp=sharing). 
- For incorporating semantic information in given embedding run:
  ```shell
  python semgcn.py -embed ./embeddings/pretrained_embed.txt 
                   -semantic synonyms -embed_dim 300 
                   -name fine_tuned_embeddings -gpu 0
  ```
* The fine-tuned embeddings will be saved in `./embeddings` directory with name `fine_tuned_embeddings`. 

### Citation:
Please cite the following paper if you use this code in your work.

```tex
@InProceedings{wordgcn2019,
  author = "Vashishth, Shikhar and Bhandari, Manik and Yadav, Prateek and Rai, Piyush and Bhattacharyya, Chiranjib and Talukdar, Partha",
  title = "Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks",
  booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  location = "Florence, Italy",
}
```
For any clarification, comments, or suggestions please create an issue or contact [shikhar@iisc.ac.in](http://shikhar-vashishth.github.io).

#include <bits/stdc++.h>
using namespace std;

FILE *fin;
unordered_map<string, int> voc2id  = unordered_map<string, int>();
unordered_map<int, string> id2voc  = unordered_map<int, string>();
unordered_map<int, int>    id2freq = unordered_map<int, int>   ();
unordered_map<string, int> de2id   = unordered_map<string, int>();

struct Entry {
	vector<int> wrds;
	vector< pair<pair<int,int>, int> > deps;
};

char *word = (char *)malloc(sizeof(char) * 1000);
int wid, offset = 0, k, idx, freq, i, j, tmp;
int num_wrds, num_deps, src, dest, lbl;
int cnt_edges, cnt_wrds, cnt_negs, cnt_sample, target, voc_size, b_elen, b_wlen;
unsigned long long next_random = (long long)1;

// Unigram distribution
const int table_size = 1e8;
int *table, train_words = 0;

void InitUnigramTable() {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < voc_size; a++) 
		train_words_pow += pow(id2freq[a], power);

	i = 0;
	d1 = pow(id2freq[i], power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(id2freq[i], power) / train_words_pow;
		}
		if (i >= voc_size) i = voc_size - 1;
	}
}

extern "C"
void init(){
	// Reading voc2id
	fin = fopen("./data/voc2id.txt", "r");
	while(fscanf(fin, "%s", word) == 1){
		tmp = fscanf(fin, "%d\n", &wid);
		voc2id[string(word)] = wid;
		id2voc[wid]	     = string(word);
	}

	voc_size = voc2id.size();

	// Reading id2freq
	fin = fopen("./data/id2freq.txt", "r");
	while(fscanf(fin, "%d", &wid) == 1){
		tmp = fscanf(fin, "%d\n", &freq);
		id2freq[wid] = freq;
		train_words += freq;
	}

	// Reading de2id
	fin = fopen("./data/de2id.txt", "r");
	while(fscanf(fin, "%s", word) == 1){
		tmp = fscanf(fin, "%d\n", &wid);
		de2id[word] = wid;
	}

	InitUnigramTable();
}


extern "C"
void reset(){
	freq = 0;
	fin  = fopen("./data/data.txt", "r");
}

int max_len = 0, cntxt_edge_label;

extern "C"
int getBatch(	int *edges, 		// Edges in the sentence graph
		int *wrds, 		// Nodes in the sentence graph
		int *neg, 		// Negative samples
		int *sub_samp, 		// Subsampling
		int *elen, 		// Edges length
		int *wlen, 		// Word length
	     	int win_size, 		// Window size for linear context
	     	int num_neg, 		// Number of negtive samples
	     	int batch_size, 	// Batchsize
	     	float sample		// Paramter for deciding rate of subsampling
	 ) {		

	cnt_edges = 0, cnt_wrds = 0, cnt_negs = 0, cnt_sample = 0;				// Count of number of edges, words, negs, samples in the entire batch

	cntxt_edge_label = de2id.size();
	
	for (int i = 0; i < batch_size; i++) {
		b_elen = 0, b_wlen = 0;								// Count of number of edges and word in particular element of batch

		if(feof(fin)) return 1;


		tmp = fscanf(fin, "%d %d", &num_wrds, &num_deps);

		for(j = 0; j < num_wrds; j++){
			tmp = fscanf(fin, "%d ", &wid);
			wrds[cnt_wrds] = wid;
			cnt_wrds++; b_wlen++;

			if (sample > 0) {							// Performing subsampling
				float ran = (sqrt(id2freq[wid] / (sample * train_words)) + 1) * (sample * train_words) / id2freq[wid];
				next_random = next_random * (unsigned long long)25214903917 + 11;
				if (ran < (next_random & 0xFFFF) / (float)65536) sub_samp[cnt_sample] = 0;
				else						 sub_samp[cnt_sample] = 1;
				cnt_sample += 1;
			}

			k = 0;
			while(k < num_neg){							// Getting negative samples
				next_random = next_random * (unsigned long long)25214903917 + 11;
				target = table[(next_random >> 16) % table_size];
				if (target == wid) continue;
				neg[cnt_negs++] = target;
				k++;
			}

		}

		for(j = 0; j < num_deps; j++){							// Including dependency edges
			tmp = fscanf(fin, "%d|%d|%d ", &src, &dest, &lbl);
			edges[cnt_edges*3 + 0] = src;
			edges[cnt_edges*3 + 1] = dest;
			edges[cnt_edges*3 + 2] = lbl;
			cnt_edges++; b_elen++;
		}

		wlen[i] = b_wlen;
		elen[i] = b_elen;

	}
	

	return 0;
}

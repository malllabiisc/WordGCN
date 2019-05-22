all:
	g++ batch_generator.cpp -o batchGen.so -fPIC -shared -pthread -O3 -march=native -std=c++11	
clean:
	rm batchGen.so
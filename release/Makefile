NVCC=nvcc -I/opt/cuda/include -O2

all: idp_g.e

idp_g.e: idp_g.o common.o answer_db.o
	g++ -L/opt/cuda/lib64 -o idp_g.e idp_g.o common.o answer_db.o -lcudart

idp_g.o: idp_g.cu
	$(NVCC) -arch compute_30 -c -o idp_g.o idp_g.cu

common.o: common.cu common.cuh
	nvcc -c -o common.o common.cu

answer_db.o: answer_db.cu answer_db.cuh
	nvcc -c -o answer_db.o answer_db.cu

clean:
	rm -f *.e *.o

SRC = $(wildcard *.c)
OBJS = $(patsubst %.c, %, $(SRC))
LIBS = -lm -lpthread
CC = mpicc

all: $(OBJS)
	
#	mpicc -o pagerank_O2O pagerank_O2O.c
#	mpicc -o pagerank_2Dpartition pagerank_2Dpartition.c -lm
#	mpicc -o pagerank_2Dpartition_sq_processors pagerank_2Dpartition_sq_processors.c -lm
#	gcc -o csr_reader csr_reader.c -lm

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf $(OBJS)

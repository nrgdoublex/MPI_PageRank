SRC = $(wildcard *.c)
OBJS = $(patsubst %.c, %, $(SRC))
LIBS = -lm -lpthread
CC = mpicc


all: $(OBJS)
	$(CC) $(CFLAGS) -o pagerank_1D_par pagerank_1D_par.o common.o $(LIBS)
	$(CC) $(CFLAGS) -o pagerank_2D_par pagerank_2D_par.o common.o $(LIBS)
	$(CC) $(CFLAGS) -o pagerank_2D_par_sqp pagerank_2D_par_sqp.o common.o $(LIBS)

%: %.c
	$(CC) $(CFLAGS) -c $< $(LIBS)

clean:
	rm -rf $(OBJS)

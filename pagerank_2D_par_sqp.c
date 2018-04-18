#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <sys/time.h>
#include <ctype.h>

#define DEBUG
#define ERROR   (0.00001f)
#define ALPHA   (0.85f)


void debug_print_int(int process, int size, int *arr);
void debug_print_double(int process, int size, double *arr);

void ReadMatrix(char *filename,int *dimension, int *num_entry,
                int **rowind, int **colind, double **values);
void SplitMatrixByRow(int dimension, int num_entry, int num_rowblocks,
                      int *from_rowind, int *from_colind, double *from_values,
                      int *value_count, int **to_rowind,
                      int **to_colind, double **to_values);
int SplitBlockByColumn( MPI_Comm *comm, int dimension, const int num_colblocks,
                        const int group, int from_size, int *from_rowind,
                        int *from_colind, double *from_values, int *to_size,
                        int **to_rowind, int **to_colind, double **to_values);
void compute_blocksize(int dimension, int rank, int num_rowblocks,
                      int num_colblocks, int *row_size, int *col_size);
void get_row_col_numblocks(int num_process, int *num_rowblocks, int *num_colblocks);


void readfile(char *filename, int *n, int *m,
              int **rowind, int **colind, double **values);
double sup_norm(int size, double *v1, double *v2);

int main(int argc, char **argv)
{
  int world_rank, world_size;
  int dimension;
  int i,j;

  /* For splitting matrix use */
  double *from_values=NULL;
	int *from_colind=NULL;
	int *from_rowind=NULL;
  double *to_values=NULL;
	int *to_colind=NULL;
	int *to_rowind=NULL;
  int num_entry, strip_size, block_size;
  int num_rowblocks, num_colblocks;


  /* For matrix multiplication use */
  int vec_size, temp_sum;
  double *x, *y, *temp, *all;
  int rowidx,colidx;
  int row_size,col_size;
  double p_sum,total_sum;
  int iter = 0;
  int stop_flag = 0;
  int *sendcounts, *displs;

  /* Data structures for timing */
	double start_time, end_time;
	struct timeval tz;
	struct timezone tx;

  /* MPI groups and communicators */
  int *root_group_idx;
  int root = 0;
  MPI_Comm row_comm, col_comm, root_comm;
  MPI_Group world_group, root_group;

  /* For reading options */
  char c, *input_name, *output_name;
  FILE *fd;

  /* Read options */
  while ((c=getopt(argc,argv,"i:o:")) != -1){
    switch (c)
    {
      case 'i':
        input_name = optarg;
        break;
      case 'o':
        output_name = optarg;
        break;
      case '?':
        if (optopt == 'i' || optopt == 'o')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,"Unknown option character `\\x%x'.\n",optopt);
        return 1;
      default:
        abort();
    }
  }

  /* Initialize the MPI environment */
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Decide split size */
  get_row_col_numblocks(world_size,&num_rowblocks,&num_colblocks);

  /* Load data from file */
  ReadMatrix(input_name,&dimension,&num_entry,
              &from_rowind,&from_colind,&from_values);


  /* Test columns sum of dataset */
#ifdef DEBUG
  // if (!world_rank){
  //   double *col_sum = (double *)malloc(sizeof(double)*dimension);
  //   int col = 0;
  //   for(i=0;i<dimension;i++)
  //     col_sum[i] = 0.0f;
  //   for (i=0;i<num_entry;i++){
  //     col_sum[from_colind[i]] += from_values[i];
  //   }
  //   for(i=0;i<dimension;i++)
  //     printf("[process %d]: column %d sum = %lf\n",world_rank,i,col_sum[i]);
  // }
  // MPI_Finalize();
#endif

  /* Create row and column groups */
  MPI_Comm_split( MPI_COMM_WORLD, world_rank / num_colblocks,
                  world_rank % num_colblocks, &row_comm);
  MPI_Comm_split( MPI_COMM_WORLD, world_rank % num_colblocks,
                  world_rank / num_colblocks, &col_comm);

  /* Create root subgroup */
  root_group_idx = (int *)malloc(sizeof(int)*num_rowblocks);
  for (i=0;i<num_rowblocks;i++)
    root_group_idx[i] = i*num_colblocks+i;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group_incl(world_group, num_rowblocks, root_group_idx, &root_group);
  MPI_Comm_create_group(MPI_COMM_WORLD, root_group, 0, &root_comm);

  /* Broadcast dimension */
  MPI_Bcast((void *) &dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Split matrix rowwisely and distribute */
  SplitMatrixByRow( dimension, num_entry, num_rowblocks,
                    from_rowind, from_colind, from_values,
                    &strip_size, &to_rowind, &to_colind, &to_values);
#ifdef DEBUG
  //debug_print_int(world_rank,strip_size,to_rowind);
  //debug_print_int(world_rank,strip_size,to_colind);
  //debug_print_double(world_rank,strip_size,to_values);
  //printf("[process %d]: number of recv values = %d\n",world_rank,strip_size);
#endif

  from_rowind = to_rowind;
  from_colind = to_colind;
  from_values = to_values;

  /* Split blocks columnwisely */
  root = SplitBlockByColumn(&row_comm, dimension, num_colblocks,
                            world_rank / num_colblocks, strip_size,
                            from_rowind, from_colind, from_values,
                            &block_size, &to_rowind, &to_colind, &to_values);
#ifdef DEBUG
  //debug_print_int(world_rank,block_size,to_rowind);
  //debug_print_int(world_rank,block_size,to_colind);
  //debug_print_double(world_rank,block_size,to_values);
  //printf("[process %d]: number of recv values = %d\n",world_rank,block_size);
#endif

  /* Compute the size of each block */
  compute_blocksize( dimension, world_rank, num_rowblocks,
                    num_colblocks, &row_size, &col_size);

  /* Initialize x,y vector */
  vec_size = ((dimension / num_colblocks) > (dimension / num_rowblocks)
              ? (dimension/num_colblocks+1)
              : (dimension/num_rowblocks+1));
  x = (double *)malloc(sizeof(double)*vec_size);
  y = (double *)malloc(sizeof(double)*vec_size);
  temp = (double *)malloc(sizeof(double)*vec_size);
  if (root){
    for (i=0;i<col_size;i++)
      x[i] = 1.0f / dimension;
  }

  /* Start measuring time */
  if (!world_rank){
    gettimeofday(&tz, &tx);
    start_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
  }

  /* PageRank main algorithm */
  do{
    if (!world_rank)
      printf("iteration = %d\n",iter);

    /* One-to-all broadcast of partial vector along process columns */
    MPI_Bcast(x,                            /* buffer */
              col_size,            /* counts */
              MPI_DOUBLE,                   /* data type */
              world_rank % num_rowblocks,   /* root */
              col_comm);                    /* communicator */

    /* Partial matrix multiplication */
    for (i=0;i<row_size;i++)
      y[i] = 0.0f;
    for (i=0;i<block_size;i++)
      y[to_rowind[i] / num_rowblocks] += to_values[i] * x[to_colind[i] / num_colblocks];

    /* All-to-one reduction along rows */
    MPI_Reduce( y, temp, row_size, MPI_DOUBLE, MPI_SUM,
                world_rank / num_colblocks, row_comm);


    if (root){

      /* Add random surfer term */
      p_sum = 0.0f;
      for (i=0;i<row_size;i++){
        temp[i] = ALPHA * temp[i];
        p_sum += temp[i];
      }
      MPI_Allreduce(&p_sum,&total_sum,1,MPI_DOUBLE,MPI_SUM,root_comm);
      total_sum = (1 - total_sum) / dimension;
      for (i=0;i<row_size;i++){
        temp[i] = temp[i] + total_sum;
      }

      /* Compare new and old vector */
      p_sum = sup_norm(row_size, x, temp);
      MPI_Reduce(&p_sum,&total_sum,1,MPI_DOUBLE,MPI_MAX,0,root_comm);
      if (!world_rank){
        //debug_print_double(world_rank,dimension,all);
        if (total_sum < ERROR)
          stop_flag = 1;
      }

      /* update x_k with x_(k+1) */
      memcpy(x,temp,sizeof(double)*(row_size));
    }

    /* Signal if it converges */
    MPI_Bcast(&stop_flag,1,MPI_INT,0,MPI_COMM_WORLD);
    if (stop_flag)
      break;

    iter += 1;
  } while (1);

  if (!world_rank)
    /* initialize data structure for sending partial vector in the future */
    all = (double *)malloc(sizeof(double)*dimension);
  if (root){
    if (!world_rank){
      sendcounts = (int *)malloc(sizeof(int)*num_rowblocks);
      displs = (int *)malloc(sizeof(int)*num_rowblocks);
      temp_sum = 0;
      for (i=0;i<num_rowblocks;i++){
        if (i < (dimension % num_rowblocks))
          sendcounts[i] = dimension / num_rowblocks + 1;
        else
          sendcounts[i] = dimension / num_rowblocks;
        displs[i] = temp_sum;
        temp_sum += sendcounts[i];
      }
    }

    /* Gather partial vector and assemble */
    MPI_Gatherv( temp,                    /* send buffer */
                row_size,      /* send counts */
                MPI_DOUBLE,             /* send data type */
                all,                      /* recv buffer */
                sendcounts,
                displs,
                MPI_DOUBLE,             /* recv data type */
                0,
                root_comm);        /* communicator */
  }

  /* Print total time elapsed */
  if (!world_rank){
    gettimeofday(&tz, &tx);
  	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
  	printf("time per message = %lf seconds\n", (end_time - start_time));
  }

  /* Output to file */
  if (!world_rank){
    fd = fopen(output_name,"w");
    fprintf(fd,"%d\n",dimension);
    for (i=0;i<dimension;i++){
      j = displs[i % num_rowblocks] + (i / num_rowblocks);
      fprintf(fd,"%lf\n",all[j]);
    }
    fclose(fd);
  }


#ifdef DEBUG
  //printf("[process %d]: number of recv values = %d\n",world_rank,block_size);
#endif

  /* release resources */
  if (!world_rank){
    free(sendcounts);
    free(displs);
    free(all);
  }
  free(root_group_idx);
  free(to_rowind);
  free(to_colind);
  free(to_values);
  free(x);
  free(y);
  free(temp);
  if (!world_rank)


  MPI_Group_free(&world_group);
  if (root){
    MPI_Group_free(&root_group);
    MPI_Comm_free(&root_comm);
  }
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  MPI_Finalize();
  return 0;

}

/*
 *  Summary:
 *      Reading the matrix from dataset
 *
 *  Input Parameters:
 *      filename:   file name of matrix dataset
 *
 *  Output Parameters:
 *      dimension : dimension of the Matrix
 *      num_entry : number of nonzero entries
 *      rowind    : array storing row indices of entries
 *      colind    : array storing column indices of entries
 *      values    : array storing entry values
 *
 */
void ReadMatrix(char *filename,int *dimension, int *num_entry,
                int **rowind, int **colind, double **values)
{
  int world_rank, world_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (!world_rank){
    /* read the whole matrix */
    readfile(filename,dimension,num_entry,rowind,colind,values);
  }
  return;
}

/*
 *  Summary:
 *      Splitting the matrix along rows and distribute to corresponding
 *      processors
 *
 *  Input Parameters:
 *      dimension : dimension of the Matrix
 *      num_entry : number of nonzero entries
 *      num_rowblocks:   number of blocks along row
 *      from_rowind:    array of row indices to be split
 *      from_colind:    array of column indices to be split
 *      from_values:    array of values to be split
 *
 *  Output Parameters:
 *      value_count:  size of array each processor gets
 *      to_rowind:    array of row indices each processor gets
 *      to_colind:    array of column indices each processor gets
 *      to_values:    array of values each processor gets
 *
 */
void SplitMatrixByRow(int dimension, int num_entry, int num_rowblocks, int *from_rowind,
                      int *from_colind, double *from_values, int *value_count,
                      int **to_rowind, int **to_colind, double **to_values)
{
  int world_rank, world_size;
  int i,j,temp_sum;
  int *sendcounts, *displs, *sendcounts_1, *displs_1, *displs_2;
  int *temp_rowind, *temp_colind;
  double *temp_values;


  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  sendcounts = (int *)malloc(sizeof(int)*world_size);
  displs = (int *)malloc(sizeof(int)*world_size);
  sendcounts_1 = (int *)malloc(sizeof(int)*world_size);
  displs_1 = (int *)malloc(sizeof(int)*world_size);
  displs_2 = (int *)malloc(sizeof(int)*world_size);
  for(i=0;i<world_size;i++)
    sendcounts[i] = displs[i] = sendcounts_1[i] = displs_1[i] = displs_2[i] = 0;

  /* Get size of blocks to distribute */
  if (!world_rank){
    for (j=0;j<num_entry;j++){
      i = from_rowind[j] % num_rowblocks;
      sendcounts[i*num_rowblocks+i] += 1;
    }
    temp_sum = 0;
    for (i=0;i<num_rowblocks;i++){
      displs[i*num_rowblocks+i] = temp_sum;
      displs_2[i*num_rowblocks+i] = displs[i*num_rowblocks+i];
      temp_sum += sendcounts[i*num_rowblocks+i];
    }
    temp_sum = 0;
    for (i=0;i<world_size;i++){
      sendcounts_1[i] = 1;
      displs_1[i] = temp_sum;
      temp_sum += sendcounts_1[i];
    }

    temp_rowind = (int *)malloc(sizeof(int)*num_entry);
    temp_colind = (int *)malloc(sizeof(int)*num_entry);
    temp_values = (double *)malloc(sizeof(double)*num_entry);

    for (j=0;j<num_entry;j++){
      i = from_rowind[j] % num_rowblocks;
      temp_rowind[displs_2[i*num_rowblocks+i]] = from_rowind[j];
      temp_colind[displs_2[i*num_rowblocks+i]] = from_colind[j];
      temp_values[displs_2[i*num_rowblocks+i]++] = from_values[j];
    }

  }

  /* We no longer need original matrix */
  if (!world_rank){
    free(from_rowind);
    free(from_colind);
    free(from_values);
  }

  /* Inform number of elements to send */
  MPI_Scatterv( sendcounts,         /* send buffer */
                sendcounts_1,       /* send counts */
                displs_1,           /* displacement */
                MPI_INT,            /* send data type */
                value_count,         /* recv buffer */
                1,                  /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */
  *to_values = (double*)malloc(sizeof(double)*(*value_count));
  *to_colind = (int*)malloc(sizeof(int)*(*value_count));
  *to_rowind = (int*)malloc(sizeof(int)*(*value_count));

  /* Split values */
  MPI_Scatterv( temp_values,            /* send buffer */
                sendcounts,       /* send counts */
                displs,           /* displacement */
                MPI_DOUBLE,       /* send data type */
                *to_values,          /* recv buffer */
                *value_count,        /* recv counts */
                MPI_DOUBLE,       /* recv data type */
                0,                /* root */
                MPI_COMM_WORLD);  /* communicator */
  /* Split colind */
  MPI_Scatterv( temp_colind,             /* send buffer */
                sendcounts,         /* send counts */
                displs,             /* displacement */
                MPI_INT,            /* send data type */
                *to_colind,           /* recv buffer */
                *value_count,          /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */
  /* Split rowind */
  MPI_Scatterv( temp_rowind,             /* send buffer */
                sendcounts,         /* send counts */
                displs,             /* displacement */
                MPI_INT,            /* send data type */
                *to_rowind,           /* recv buffer */
                *value_count,          /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */

  /* free temporary data structure */
  if (!world_rank){
    free(temp_rowind);
    free(temp_colind);
    free(temp_values);
  }
  free(sendcounts);
  free(displs);
  free(sendcounts_1);
  free(displs_1);
  free(displs_2);
  return;
}

/*
 *  Summary:
 *      Splitting the row blocks along columns and distribute to corresponding
 *      processors
 *
 *  Input Parameters:
 *      comm:           communicator of each row group
 *      dimension:      dimension of the Matrix
 *      num_colblocks:  number of blocks along column
 *      group:          group number of row group
 *      from_size:      size of from_rowind, from_colind and from_values array
 *      from_rowind:    array of row indices to be split
 *      from_colind:    array of column indices to be split
 *      from_values:    array of values to be split
 *
 *  Output Parameters:
 *      to_size:    size of array each processor gets
 *      to_rowind:  array of row indices each processor gets
 *      to_colind:  array of column indices each processor gets
 *      to_values:  array of values each processor gets
 *
 *  Return:
 *      1 if current processor is the head of row(column) groups,
 *      0 otherwise
 *
 */
int SplitBlockByColumn(MPI_Comm *comm, int dimension, const int num_colblocks,
                        const int group, int from_size,
                        int *from_rowind, int *from_colind, double *from_values,
                        int *to_size, int **to_rowind, int **to_colind, double **to_values)
{
  int world_rank, world_size;
  int i;
  int *buf_rowind, *buf_colind;
  double *buf_values;
  int block_size[num_colblocks];
  int idx_set[num_colblocks];
  int sendcounts[num_colblocks];
  int displs[num_colblocks];
  int idx;
  int total;
  int root = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Recognize head of group */
  if (group == world_rank % num_colblocks)
    root = 1;

  if (root){
    for (i=0;i<num_colblocks;i++){
      block_size[i] = 0;
      sendcounts[i] = 1;
      displs[i] = i;
    }

    for (i=0;i<from_size;i++){
      idx = from_colind[i] % num_colblocks;
      block_size[idx] += 1;
    }

    total = block_size[0];
    idx_set[0] = 0;
    for (i=1;i<num_colblocks;i++){
      total += block_size[i];
      idx_set[i] = total - block_size[i];
    }

    /* rearrange values,rowind,colind */
    buf_values = (double *)malloc(sizeof(double)*from_size);
    buf_rowind = (int *)malloc(sizeof(int)*from_size);
    buf_colind = (int *)malloc(sizeof(int)*from_size);
    for (i=0;i<from_size;i++){
      idx = from_colind[i] % num_colblocks;
      buf_values[idx_set[idx]] = from_values[i];
      buf_rowind[idx_set[idx]] = from_rowind[i];
      buf_colind[idx_set[idx]++] = from_colind[i];
    }
  }

  /* notify each process the size to receive */
  MPI_Scatterv( block_size,         /* send buffer */
                sendcounts,       /* send counts */
                displs,           /* displacement */
                MPI_INT,            /* send data type */
                to_size,         /* recv buffer */
                1,                  /* recv counts */
                MPI_INT,            /* recv data type */
                group,                  /* root */
                *comm);    /* communicator */

  (*to_rowind) = (int *)malloc(sizeof(int)*(*to_size));
  (*to_colind) = (int *)malloc(sizeof(int)*(*to_size));
  (*to_values) = (double *)malloc(sizeof(double)*(*to_size));
  total = block_size[0];
  idx_set[0] = 0;
  for (i=1;i<num_colblocks;i++){
    total += block_size[i];
    idx_set[i] = total - block_size[i];
  }
  /* split rowind */
  MPI_Scatterv( buf_rowind,             /* send buffer */
                block_size,         /* send counts */
                idx_set,             /* displacement */
                MPI_INT,            /* send data type */
                *to_rowind,           /* recv buffer */
                *to_size,          /* recv counts */
                MPI_INT,            /* recv data type */
                group,                  /* root */
                *comm);    /* communicator */
  /* split colind */
  MPI_Scatterv( buf_colind,             /* send buffer */
                block_size,         /* send counts */
                idx_set,             /* displacement */
                MPI_INT,            /* send data type */
                *to_colind,           /* recv buffer */
                *to_size,          /* recv counts */
                MPI_INT,            /* recv data type */
                group,                  /* root */
                *comm);    /* communicator */
  /* split values */
  MPI_Scatterv( buf_values,             /* send buffer */
                block_size,         /* send counts */
                idx_set,             /* displacement */
                MPI_DOUBLE,            /* send data type */
                *to_values,           /* recv buffer */
                *to_size,          /* recv counts */
                MPI_DOUBLE,            /* recv data type */
                group,                  /* root */
                *comm);    /* communicator */

  if (root){
    free(buf_values);
    free(buf_rowind);
    free(buf_colind);
  }
  if (!world_rank){
    free(from_rowind);
    free(from_colind);
    free(from_values);
  }
  return root;
}

/*
 *  Summary:
 *      Compute the size of block each processor is responsible for
 *
 *  Input Parameters:
 *      dimension:  dimension of the Matrix
 *      rank:       processor id
 *      num_rowblocks:  number of blocks along row
 *      num_colblocks:  number of blocks along column
 *
 *  Output Parameters:
 *      row_size:   size of block along row
 *      col_size:   size of block along column
 *
 */
void compute_blocksize( int dimension, int rank, int num_rowblocks,
                        int num_colblocks, int *row_size, int *col_size)
{
  int i;

  *row_size = *col_size = 0;

  /* compute size along row */
  i = rank / num_rowblocks;
  if (i < (dimension % num_rowblocks))
    *row_size = dimension / num_rowblocks + 1;
  else
    *row_size = dimension / num_rowblocks;

  /* compute size along column */
  i = rank % num_colblocks;
  if (i < (dimension % num_colblocks))
    *col_size = dimension / num_colblocks + 1;
  else
    *col_size = dimension / num_colblocks;
}

/*
 *  Summary:
 *      Implementation of reading matrix from files
 *
 *  Input Parameters:
 *      filename:  file name
 *
 *  Output Parameters:
 *      n:      dimension of the matrix
 *      m:      number of nonzero entries
 *      rowind: array of row indices
 *      colind: array of column indices
 *      values: array of values
 *
 */
void readfile(char *filename, int *n, int *m,
              int **rowind, int **colind, double **values)
{
	FILE *fp=fopen(filename,"r");
	char buf[200];
	int i,j,k,l;
	int p,q;
	double w;
  int previous_row = 0;
  int splitter_idx = 0;
	if ((fp=fopen(filename,"r"))==NULL){
	  fprintf(stderr,"Open file errer, check filename please.\n");
	}

  /* matrix dimension and number of nonzero entries */
	fgets(buf,200,fp);
	*n=atoi(buf);
	l=0;
	while(buf[l++]!=' ');
	*m=atoi(&buf[l]);
	printf("Matrix size: %d, #Non-zeros: %d\n",*n,*m);

  /* sparse matrix */
	(*rowind)=(int*)malloc(sizeof(int)*(*m));
	(*colind)=(int*)malloc(sizeof(int)*(*m));
	(*values)=(double*)malloc(sizeof(double)*(*m));
	k=-1;
	for(i=0;i<(*m);i++){
	  fgets(buf,200,fp);
	  l=0;p=atoi(&buf[l]);
	  while(buf[l++]!=' '); q=atoi(&buf[l]);
	  while(buf[l++]!=' '); w=atof(&buf[l]);

	  (*values)[i]=w;
	  (*colind)[i]=q;
	  (*rowind)[i]=p;
	}

	fclose(fp);
}

/*
 *  Summary:
 *      Compute the sup norm of 2 vectors
 *
 *  Input Parameters:
 *      size: size of vectors
 *      v1:   array1
 *      v2:   array2
 *
 *  Return:
 *      sup norm of array1 and array2
 *
 */
double sup_norm(int size, double *v1, double *v2)
{
  int i;
  double diff;
  double ret = 0.0f;
  for (i=0;i<size;i++){
    diff = v1[i] - v2[i];
    diff = ((diff > 0) - (diff < 0)) * diff;
    if (diff > ret)
      ret = diff;
  }
  return ret;
}

/*
 *  Summary:
 *      Print values of vector
 *
 *  Input Parameters:
 *      process:  processor id
 *      size:     size of array
 *      arr:      array to print
 *
 */
void debug_print_int(int process, int size, int *arr)
{
  int i;
  char output[256], *pos = output;
  pos += sprintf(pos,"[process %d]: ", process);
  for (i=0;i<size;i++){
    pos += sprintf(pos,"%d ",arr[i]);
  }
  pos += sprintf(pos,"\n");
  printf("%s",output);
}

/*
 *  Summary:
 *      Print values of vector
 *
 *  Input Parameters:
 *      process:  processor id
 *      size:     size of array
 *      arr:      array to print
 *
 */
void debug_print_double(int process, int size, double *arr)
{
  int i;
  char output[256], *pos = output;
  pos += sprintf(pos,"[process %d]: ", process);
  for (i=0;i<size;i++){
    pos += sprintf(pos,"%lf ",arr[i]);
  }
  pos += sprintf(pos,"\n");
  printf("%s",output);
}

void get_row_col_numblocks(int num_process, int *num_rowblocks, int *num_colblocks)
{
  int idx = (int) sqrt((double) num_process);
  while (idx > 0 && num_process % idx)
    idx -= 1;
  *num_colblocks = idx;
  *num_rowblocks = num_process / idx;
  return;
}

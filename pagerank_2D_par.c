#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <sys/time.h>
#include <ctype.h>
#include "common.h"

//#define DEBUG
#define ERROR   (0.00001f)
#define ALPHA   (0.85f)

void ReadMatrix(char *filename,int *dimension, int *num_entry,
                int **rowind, int **colind, double **values, int verbose);
void SplitMatrixByRow(int dimension, int num_entry, int num_rowblocks,
                      int num_colblocks, int *from_rowind,
                      int *from_colind, double *from_values, int *value_count,
                      int **to_rowind, int **to_colind, double **to_values);
int SplitBlockByColumn( MPI_Comm *comm, int dimension, const int num_colblocks,
                        const int group, int from_size, int *from_rowind,
                        int *from_colind, double *from_values, int *to_size,
                        int **to_rowind, int **to_colind, double **to_values);
void compute_blocksize(int dimension, int rank, int num_rowblocks,
                      int num_colblocks, int *row_size, int *col_size);
void get_row_col_numblocks(int num_process, int *num_rowblocks, int *num_colblocks);

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
  double *x, *y, *re_y, *px, *py, *temp;
  int rowidx,colidx;
  int row_size,col_size;
  double p_sum,total_sum;
  int iter = 0;
  int stop_flag = 0;
  int *rowwise_sendcounts, *rowwise_displs, *colwise_sendcounts, *colwise_displs;

  /* MPI groups and communicators */
  int *row_root_group_idx, *col_root_group_idx;
  int root = 0;
  MPI_Comm row_comm, col_comm, row_root_comm, col_root_comm;
  MPI_Group world_group, row_root_group, col_root_group;

  /* Options use */
  Options *options;
  FILE *fd;

  /* Initialize the MPI environment */
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Read options */
  init_options(&options);
  if (!world_rank){
    if (read_options(argc,argv,options)){
      MPI_Abort(MPI_COMM_WORLD,1);
      return -1;
    }
  }

  /* Decide split size */
  get_row_col_numblocks(world_size,&num_rowblocks,&num_colblocks);

  /* Load data from file */
  ReadMatrix(options->input_name,&dimension,&num_entry,
              &from_rowind,&from_colind,&from_values,options->verbose);

  /* Create row and column groups */
  MPI_Comm_split( MPI_COMM_WORLD, world_rank / num_colblocks,
                  world_rank % num_colblocks, &row_comm);
  MPI_Comm_split( MPI_COMM_WORLD, world_rank % num_colblocks,
                  world_rank / num_colblocks, &col_comm);

  /* Create root subgroup */
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  row_root_group_idx = (int *)malloc(sizeof(int)*num_rowblocks);
  for (i=0;i<num_rowblocks;i++)
    row_root_group_idx[i] = i*num_colblocks;
  MPI_Group_incl(world_group, num_rowblocks, row_root_group_idx, &row_root_group);
  MPI_Comm_create_group(MPI_COMM_WORLD, row_root_group, 0, &row_root_comm);

  col_root_group_idx = (int *)malloc(sizeof(int)*num_colblocks);
  for (i=0;i<num_colblocks;i++)
    col_root_group_idx[i] = i;
  MPI_Group_incl(world_group, num_colblocks, col_root_group_idx, &col_root_group);
  MPI_Comm_create_group(MPI_COMM_WORLD, col_root_group, 0, &col_root_comm);

  /* Broadcast dimension */
  MPI_Bcast((void *) &dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Split matrix rowwisely and distribute */
  SplitMatrixByRow( dimension, num_entry, num_rowblocks, num_colblocks,
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
                            (world_rank / num_colblocks), strip_size,
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

  /* Initialize x,y vector and necessary data structures */
  /* during multiplication                               */
  vec_size = ((dimension / num_colblocks) > (dimension / num_rowblocks)
              ? (dimension/num_colblocks+1)
              : (dimension/num_rowblocks+1));
  px = (double *)malloc(sizeof(double)*vec_size);
  py = (double *)malloc(sizeof(double)*vec_size);
  temp = (double *)malloc(sizeof(double)*vec_size);
  if (!world_rank){
    rowwise_sendcounts = (int *)malloc(sizeof(int)*num_rowblocks);
    rowwise_displs = (int *)malloc(sizeof(int)*num_rowblocks);
    temp_sum = 0;
    for (i=0;i<num_rowblocks;i++){
      if (i < (dimension % num_rowblocks))
        rowwise_sendcounts[i] = dimension / num_rowblocks + 1;
      else
        rowwise_sendcounts[i] = dimension / num_rowblocks;
      rowwise_displs[i] = temp_sum;
      temp_sum += rowwise_sendcounts[i];
    }

    colwise_sendcounts = (int *)malloc(sizeof(int)*num_colblocks);
    colwise_displs = (int *)malloc(sizeof(int)*num_colblocks);
    temp_sum = 0;
    for (i=0;i<num_colblocks;i++){
      if (i < (dimension % num_colblocks))
        colwise_sendcounts[i] = dimension / num_colblocks + 1;
      else
        colwise_sendcounts[i] = dimension / num_colblocks;
      colwise_displs[i] = temp_sum;
      temp_sum += colwise_sendcounts[i];
    }


    x = (double *)malloc(sizeof(double)*dimension);
    y = (double *)malloc(sizeof(double)*dimension);
    re_y = (double *)malloc(sizeof(double)*dimension);
    for (i=0;i<dimension;i++)
      x[i] = 1.0f / dimension;
  }

  /* Start measuring time */
  if (!world_rank){
    gettimeofday(&tz, &tx);
    start_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
  }

  /* PageRank main algorithm */
  iter = 1;
  do{
    if (!world_rank && options->verbose)
      printf("iteration %d\n",iter);


    /* Split the vector into blocks */
    if (world_rank < num_colblocks){
      MPI_Scatterv( x,                  /* send buffer */
                    colwise_sendcounts,       /* send counts */
                    colwise_displs,           /* displacement */
                    MPI_DOUBLE,       /* send data type */
                    px,          /* recv buffer */
                    col_size,        /* recv counts */
                    MPI_DOUBLE,       /* recv data type */
                    0,                /* root */
                    col_root_comm);  /* communicator */
    }



    /* One-to-all broadcast of partial vector along process columns */
    MPI_Bcast(px,                            /* buffer */
              col_size,            /* counts */
              MPI_DOUBLE,                   /* data type */
              0,   /* root */
              col_comm);                    /* communicator */

    /* Partial matrix multiplication */
    for (i=0;i<row_size;i++)
      py[i] = 0.0f;
    for (i=0;i<block_size;i++)
      py[to_rowind[i] / num_rowblocks] += to_values[i] * px[to_colind[i] / num_colblocks];

    /* All-to-one reduction along rows */
    MPI_Reduce( py, temp, row_size, MPI_DOUBLE, MPI_SUM, 0, row_comm);


    if (root){
      /* Add random surfer term */
      p_sum = 0.0f;
      for (i=0;i<row_size;i++){
        temp[i] = ALPHA * temp[i];
        p_sum += temp[i];
      }
      MPI_Allreduce(&p_sum,&total_sum,1,MPI_DOUBLE,MPI_SUM,row_root_comm);
      total_sum = (1 - total_sum) / dimension;
      for (i=0;i<row_size;i++)
        temp[i] += total_sum;


      /* Gather partial vector and assemble */
      MPI_Gatherv(temp,                    /* send buffer */
                  row_size,      /* send counts */
                  MPI_DOUBLE,             /* send data type */
                  re_y,                      /* recv buffer */
                  rowwise_sendcounts,
                  rowwise_displs,
                  MPI_DOUBLE,             /* recv data type */
                  0,
                  row_root_comm);        /* communicator */

      if (!world_rank){
        /* rearrange array */
        for (i=0;i<dimension;i++){
          y[colwise_displs[i % num_colblocks] + (i / num_colblocks)] = \
          re_y[rowwise_displs[i % num_rowblocks] + (i / num_rowblocks)];
        }

        /* If convergence, quit */
        total_sum = sup_norm(dimension,x,y);
        if (total_sum < ERROR)
          stop_flag = 1;

        /* If overtime, quit */
        gettimeofday(&tz, &tx);
        end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
        if (options->time_limit > 0 && options->time_limit <= (end_time-start_time))
          stop_flag = 1;

        /* If max iteration reached, quit */
        if (options->max_iter > 0 && options->max_iter == iter)
          stop_flag = 1;

        memcpy(x,y,sizeof(double)*dimension);
      }
    }

    /* Signal if it converges */
    MPI_Bcast(&stop_flag,1,MPI_INT,0,MPI_COMM_WORLD);
    if (stop_flag)
      break;

    iter += 1;
  } while (1);


  /* Print total time elapsed */
  if (!world_rank && options->verbose)
    printf("time elapsed = %lf seconds\n", (end_time - start_time));

  /* Output to file or printout */
  if (!world_rank){
    if (options->output_name){
      fd = fopen(options->output_name,"w");
      fprintf(fd,"%d\n",dimension);
      for (i=0;i<dimension;i++){
        j = colwise_displs[i % num_colblocks] + (i / num_colblocks);
        fprintf(fd,"%lf\n",x[j]);
      }
      fclose(fd);
    }
    else{
      printf("The resulting vector is:\n");
      for (i=0;i<dimension;i++){
        j = colwise_displs[i % num_colblocks] + (i / num_colblocks);
        printf("%lf\n",x[j]);
      }
    }
  }


#ifdef DEBUG
  //printf("[process %d]: number of recv values = %d\n",world_rank,block_size);
#endif

  /* release resources */
  if (!world_rank){
    free(rowwise_sendcounts);
    free(rowwise_displs);
    free(colwise_sendcounts);
    free(colwise_displs);
    free(x);
    free(y);
    free(re_y);
  }
  free(row_root_group_idx);
  free(col_root_group_idx);
  free(to_rowind);
  free(to_colind);
  free(to_values);
  free(px);
  free(py);
  free(temp);

  MPI_Group_free(&world_group);
  if (root){
    MPI_Group_free(&row_root_group);
    MPI_Comm_free(&row_root_comm);
  }
  if (world_rank < num_colblocks){
    MPI_Group_free(&col_root_group);
    MPI_Comm_free(&col_root_comm);
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
                int **rowind, int **colind, double **values, int verbose)
{
  int world_rank, world_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (!world_rank){
    /* read the whole matrix */
    get_matrix(filename,dimension,num_entry,rowind,colind,values,verbose);
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
void SplitMatrixByRow(int dimension, int num_entry, int num_rowblocks,
                      int num_colblocks, int *from_rowind,
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
      sendcounts[i*num_colblocks] += 1;
    }
    temp_sum = 0;
    for (i=0;i<num_rowblocks;i++){
      displs[i*num_colblocks] = temp_sum;
      displs_2[i*num_colblocks] = displs[i*num_colblocks];
      temp_sum += sendcounts[i*num_colblocks];
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
      temp_rowind[displs_2[i*num_colblocks]] = from_rowind[j];
      temp_colind[displs_2[i*num_colblocks]] = from_colind[j];
      temp_values[displs_2[i*num_colblocks]++] = from_values[j];
    }

    /* We no longer need original matrix */
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
  if (!(world_rank % num_colblocks))
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

    /* We no longer need them */
    free(from_rowind);
    free(from_colind);
    free(from_values);
  }

  /* notify each process the size to receive */
  MPI_Scatterv( block_size,         /* send buffer */
                sendcounts,       /* send counts */
                displs,           /* displacement */
                MPI_INT,            /* send data type */
                to_size,         /* recv buffer */
                1,                  /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
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
                0,                  /* root */
                *comm);    /* communicator */
  /* split colind */
  MPI_Scatterv( buf_colind,             /* send buffer */
                block_size,         /* send counts */
                idx_set,             /* displacement */
                MPI_INT,            /* send data type */
                *to_colind,           /* recv buffer */
                *to_size,          /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                *comm);    /* communicator */
  /* split values */
  MPI_Scatterv( buf_values,             /* send buffer */
                block_size,         /* send counts */
                idx_set,             /* displacement */
                MPI_DOUBLE,            /* send data type */
                *to_values,           /* recv buffer */
                *to_size,          /* recv counts */
                MPI_DOUBLE,            /* recv data type */
                0,                  /* root */
                *comm);    /* communicator */

  if (root){
    free(buf_values);
    free(buf_rowind);
    free(buf_colind);
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
  i = rank / num_colblocks;
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
 *      Decide number of row and column blocks to split
 *
 *  Input Parameters:
 *      num_process:    number of processors
 *
 *  Output Parameters:
 *      num_rowblocks:  number of blocks along row
 *      num_colblocks:  number of blocks along column
 *
 */
void get_row_col_numblocks(int num_process, int *num_rowblocks, int *num_colblocks)
{
  int idx = (int) sqrt((double) num_process);
  while (idx > 0 && num_process % idx)
    idx -= 1;
  *num_colblocks = idx;
  *num_rowblocks = num_process / idx;
  return;
}

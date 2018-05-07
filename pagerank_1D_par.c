#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include <ctype.h>
#include <unistd.h>
#include "common.h"

#define DEBUG
#define ERROR (0.00001f)
#define ALPHA (0.85f)

void csr_multiply(int n, double *value, int *colind, int *rbegin, double *x, double **answer);
void gen_sendcounts_displs(int dimension, int num_process, int *rbegin, int *sendcounts,
                            int *displs, int process_id, int rbegin_flag);

int main(int argc, char **argv)
{
  /* Global variable */
  int dimension;
  int stop_flag = 0;

  /* For splitting matrix use */
  double *value=NULL;
	int *colind=NULL;
	int *rbegin=NULL;
  double *value_p=NULL;
	int *colind_p=NULL;
	int *rbegin_p=NULL;
  int num_rows;
  int num_nonzero;
  int *sendcounts, *displs;
  int recvcount;
  int temp_sum;
  int i,iter;
  double p_sum, total_sum;
  double *x_p,*x,*y;

  FILE *fd;
  Options *options;

  /* Initialize the MPI environment */
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Read options */
  init_options(&options);
  if (!world_rank){
    if (read_options(argc,argv,options)){
      MPI_Abort(MPI_COMM_WORLD,1);
      return -1;
    }
  }

  /* Read matrix from file */
  if (!world_rank){
    /* read the whole matrix */
    get_sparse_matrix(options->input_name, &dimension, &num_nonzero,
                      &value, &colind, &rbegin, options->verbose);
  }

  /* Prepare to split */
  sendcounts = (int *)malloc(sizeof(int)*world_size);
  displs = (int *)malloc(sizeof(int)*world_size);
  gen_sendcounts_displs(dimension,world_size,rbegin,sendcounts,displs,world_rank,1);


  /* Broadcast dimension */
  MPI_Bcast((void *) &dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Notify size of each block */
  int *scnt_sendcounts = (int *)malloc(sizeof(int)*world_size);
  int *scnt_displs = (int *)malloc(sizeof(int)*world_size);
  temp_sum = 0;
  for (i=0;i<world_size;i++){
    scnt_sendcounts[i] = 1;
    scnt_displs[i] = temp_sum;
    temp_sum += scnt_sendcounts[i];
  }
  MPI_Scatterv( sendcounts,         /* send buffer */
                scnt_sendcounts,       /* send counts */
                scnt_displs,           /* displacement */
                MPI_INT,            /* send data type */
                &recvcount,         /* recv buffer */
                1,                  /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */
  free(scnt_sendcounts);
  free(scnt_displs);

  /* Split the matrix */
  value_p = (double *)malloc(sizeof(double)*recvcount);
  MPI_Scatterv( value,            /* send buffer */
                sendcounts,       /* send counts */
                displs,           /* displacement */
                MPI_DOUBLE,       /* send data type */
                value_p,          /* recv buffer */
                recvcount,        /* recv counts */
                MPI_DOUBLE,       /* recv data type */
                0,                /* root */
                MPI_COMM_WORLD);  /* communicator */
  colind_p = (int *)malloc(sizeof(int)*recvcount);
  MPI_Scatterv( colind,             /* send buffer */
                sendcounts,         /* send counts */
                displs,             /* displacement */
                MPI_INT,            /* send data type */
                colind_p,           /* recv buffer */
                recvcount,          /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */


  gen_sendcounts_displs(dimension,world_size,rbegin,sendcounts,displs,world_rank,0);

  /* Size of rbegin subarray = 1 + number of rows */
  num_rows = sendcounts[world_rank] - 1;
  rbegin_p = (int *)malloc(sizeof(int)*(num_rows+1));
  MPI_Scatterv( rbegin,             /* send buffer */
                sendcounts,         /* send counts */
                displs,             /* displacement */
                MPI_INT,            /* send data type */
                rbegin_p,           /* recv buffer */
                (num_rows+1),          /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */

  /* Prepare multiplication */
  x_p = (double *)malloc(sizeof(double)*num_rows);
  x = (double *)malloc(sizeof(double)*dimension);
  if (!world_rank){
    y = (double *)malloc(sizeof(double)*dimension);
  }

  for (i=0;i<dimension;i++)
    x[i] = 0.0f;
  for (i=0;i<num_rows;i++)
    x_p[i] = 1.0f / dimension;
  for (i=0;i<world_size;i++)
    sendcounts[i] = sendcounts[i] - 1;


  /* Start measuring time */
  if (!world_rank){
    gettimeofday(&tz, &tx);
  	start_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
  }

  iter = 1;
  do {
    if(!world_rank && options->verbose)
      printf("iteration %d\n", iter);

    /* All-to-all broadcast */
    MPI_Allgatherv( x_p,                    /* send buffer */
                    num_rows,      /* send counts */
                    MPI_DOUBLE,             /* send data type */
                    x,                      /* recv buffer */
                    sendcounts,             /* recv counts */
                    displs,
                    MPI_DOUBLE,             /* recv data type */
                    MPI_COMM_WORLD);        /* communicator */

    /* y = alpha * P_link * x_k */
    csr_multiply(num_rows, value_p, colind_p, rbegin_p, x, &x_p);
    p_sum = 0.0f;
    for (i=0;i<num_rows;i++){
      x_p[i] = ALPHA * x_p[i];
      p_sum += x_p[i];
    }

    /* Reduce sum of y, and then update by adding random surfer term */
    MPI_Allreduce(&p_sum,&total_sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    total_sum = (1 - total_sum) / dimension;
    for (i=0;i<num_rows;i++)
      x_p[i] = x_p[i] + total_sum;

    /* Assemble the resulting vector */
    MPI_Gatherv(  x_p,                    /* send buffer */
                  num_rows,               /* send counts */
                  MPI_DOUBLE,             /* send data type */
                  y,                      /* recv buffer */
                  sendcounts,             /* send counts */
                  displs,                 /* recv counts */
                  MPI_DOUBLE,             /* recv data type */
                  0,                      /* root */
                  MPI_COMM_WORLD);        /* communicator */

    if (!world_rank){
      gettimeofday(&tz, &tx);
      end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
    }

    /* Leave if converge */
    if (!world_rank){
      if (sup_norm(dimension,x,y) < ERROR)
        stop_flag = 1;
      if (options->max_iter > 0 && options->max_iter == iter)
        stop_flag = 1;
      if (options->time_limit > 0 && (end_time-start_time) >= options->time_limit)
        stop_flag = 1;
    }

    /* Notify if we are finished */
    MPI_Bcast(&stop_flag,1,MPI_INT,0,MPI_COMM_WORLD);
    if (stop_flag)
      break;

    iter += 1;
  } while (1);

  /*Pprint total elapsed time */
  if (!world_rank && options->verbose)
    printf("time elapsed = %lf seconds\n", (end_time - start_time));


  if (!world_rank){
    /* Output to file */
    if (options->output_name){
      fd = fopen(options->output_name,"w");
      fprintf(fd,"%d\n",dimension);
      for (i=0;i<dimension;i++){
        fprintf(fd,"%lf\n",y[i]);
      }
      fclose(fd);
    }
    else{
      /* Print the vector */
      printf("The resulting vector is:\n");
      for (i=0;i<dimension;i++){
        printf("%lf\n",y[i]);
      }
    }
  }

  /* Free memory */
  free(x_p);
  free(x);
  if (!world_rank){
    free(y);
  }
  free(sendcounts);
  free(displs);
  free(value_p);
  free(colind_p);
  free(rbegin_p);
  free(options);

  MPI_Finalize();

  return 0;
}

/*
 *  Summary:
 *      Perform Sparse Matrix-Vector Multiplication
 *
 *  Input Parameters:
 *      n:  dimension of sparse matrix
 *      value:  array of nonzero values of sparse matrix
 *      colind:  array of column indices of nonzero values of sparse matrix
 *      rbegin:  array of number of nonzero values of sparse matrix before row[i]
 *
 *  Output Parameters:
 *      options:  Options data structure
 *
 *  Return:
 *      0 if success, 1 otherwise
 *
 */
void csr_multiply(int n, double *value, int *colind, int *rbegin, double *x, double **answer)
{
	int i,k1,k2,k,j, k_start;
	double *temp = *answer;

  k_start = rbegin[0];
	for (i=0;i<n;i++){
		temp[i] = 0;
		k1 = rbegin[i];
		k2 = rbegin[i+1] - 1;
		if (k1 > k2)
			continue;
		for (k=k1;k<=k2;k++){
			j = colind[k-k_start];
			temp[i] += value[k-k_start]*x[j];
		}
	}
	return;
}

/*
 *  Summary:
 *      Generate sendcounts and displacement array
 *
 *  Input Parameters:
 *      dimension:    dimension of sparse matrix
 *      num_process:  number of processes
 *      rbegin:       rbegin array
 *      process_id:   current process id
 *      rbegin_flag:  flag to indicate the sendcounts and displs arrries are for rbegin array
 *
 *  Output Parameters:
 *      sendcounts:   sendcounts array used in MPI interface
 *      displs:       displacement array used in MPI interface
 *
 */
void gen_sendcounts_displs(int dimension, int num_process, int *rbegin, int *sendcounts,
                            int *displs, int process_id, int rbegin_flag)
{
  int temp_sum = 0;
  int idx_start,idx_end;
  int i;
  for (i=0;i<num_process;i++){
    if (i < (dimension % num_process)){
      idx_start = (dimension / num_process + 1) * i;
      idx_end = (dimension / num_process + 1) * (i + 1);
    }
    else{
      idx_start = (dimension / num_process + 1) * i - (i - dimension % num_process);
      idx_end = (dimension / num_process + 1) * (i + 1) - (i - dimension % num_process + 1);
    }
    /* for process 0, no need to broadcast */
    if (rbegin_flag){
      if (!process_id){
        idx_start = rbegin[idx_start];
        idx_end = rbegin[idx_end];
      }
      sendcounts[i] = idx_end - idx_start;
      displs[i] = temp_sum;
      temp_sum += sendcounts[i];
    }
    else{
      sendcounts[i] = idx_end - idx_start + 1;
      displs[i] = temp_sum;
      temp_sum += sendcounts[i] - 1;
    }
  }
}

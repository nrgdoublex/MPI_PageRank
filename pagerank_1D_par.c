#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include <ctype.h>
#include <unistd.h>

#define DEBUG
#define ERROR (0.00001f)
#define ALPHA (0.85f)


void readfile(char *fileName, int *n, int *m, double **value, int **colind, int **rbegin);
void output(int n, double *answer);
void csr_multiply(int n, double *value, int *colind, int *rbegin, double *x, double **answer);
double sup_norm(int size, double *v1, double *v2);

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
  int block_size;
  int row_start, row_end;
  int num_nonzero;
  int *sendcounts, *displs;
  int recvcount;
  int temp_sum;
  int i,iter;
  int idx_start, idx_end;
  double p_sum, total_sum;

  /* Data structures for timing */
  double start_time, end_time;
  struct timeval tz;
  struct timezone tx;

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
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* Read matrix from file */
  if (!world_rank){
    /* read the whole matrix */
    readfile(input_name,&dimension,&num_nonzero,&value,&colind,&rbegin);
  }

  /* Prepare to split */
  sendcounts = (int *)malloc(sizeof(int)*world_size);
  displs = (int *)malloc(sizeof(int)*world_size);
  temp_sum = 0;
  for (i=0;i<world_size;i++){
    if (i < (dimension % world_size)){
      idx_start = (dimension / world_size + 1) * i;
      idx_end = (dimension / world_size + 1) * (i + 1);
    }
    else{
      idx_start = (dimension / world_size + 1) * i - (i - dimension % world_size);
      idx_end = (dimension / world_size + 1) * (i + 1) - (i - dimension % world_size + 1);
    }
    /* for process 0, we need to broadcast value and colind, so use different sendcounts */
    if (!world_rank){
      idx_start = rbegin[idx_start];
      idx_end = rbegin[idx_end];
    }
    sendcounts[i] = idx_end - idx_start;
    displs[i] = temp_sum;
    temp_sum += sendcounts[i];
  }

  /* Broadcast dimension */
  MPI_Bcast((void *) &dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* Notify size of each block */
  int *sendcounts_1 = (int *)malloc(sizeof(int)*world_size);
  int *displs_1 = (int *)malloc(sizeof(int)*world_size);
  temp_sum = 0;
  for (i=0;i<world_size;i++){
    sendcounts_1[i] = 1;
    displs_1[i] = temp_sum;
    temp_sum += sendcounts_1[i];
  }
  MPI_Scatterv( sendcounts,         /* send buffer */
                sendcounts_1,       /* send counts */
                displs_1,           /* displacement */
                MPI_INT,            /* send data type */
                &recvcount,         /* recv buffer */
                1,                  /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */
  free(sendcounts_1);
  free(displs_1);

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

  temp_sum = 0;
  for (i=0;i<world_size;i++){
    if (i < (dimension % world_size)){
      idx_start = (dimension / world_size + 1) * i;
      idx_end = (dimension / world_size + 1) * (i + 1);
    }
    else{
      idx_start = (dimension / world_size + 1) * i - (i - dimension % world_size);
      idx_end = (dimension / world_size + 1) * (i + 1) - (i - dimension % world_size + 1);
    }
    sendcounts[i] = idx_end - idx_start + 1;
    displs[i] = temp_sum;
    temp_sum += idx_end - idx_start;
  }
  row_start = displs[world_rank];
  row_end = displs[world_rank] + sendcounts[world_rank] - 1;
  rbegin_p = (int *)malloc(sizeof(int)*(row_end-row_start+1));
  MPI_Scatterv( rbegin,             /* send buffer */
                sendcounts,         /* send counts */
                displs,             /* displacement */
                MPI_INT,            /* send data type */
                rbegin_p,           /* recv buffer */
                (row_end-row_start+1),          /* recv counts */
                MPI_INT,            /* recv data type */
                0,                  /* root */
                MPI_COMM_WORLD);    /* communicator */

  /* Prepare multiplication */
  double *x_p = (double *)malloc(sizeof(double)*(row_end-row_start));
  double *x = (double *)malloc(sizeof(double)*dimension);
  double *y = (double *)malloc(sizeof(double)*dimension);
  for (i=0;i<dimension;i++)
    x[i] = y[i] = 0.0f;
  for (i=0;i<(row_end-row_start);i++)
    x_p[i] = 1.0f / dimension;
  for (i=0;i<world_size;i++)
    sendcounts[i] = sendcounts[i] - 1;
  int recvcounts_2[world_size];
  for (i=0;i<world_size;i++)
    recvcounts_2[i] = 1;
  iter = 0;

  /* Start measuring time */
  if (!world_rank){
    gettimeofday(&tz, &tx);
  	start_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
  }

  do {
    if(!world_rank)
      printf("iteration = %d\n", iter);

    /* All-to-all broadcast */
    MPI_Allgatherv( x_p,                    /* send buffer */
                    row_end-row_start,      /* send counts */
                    MPI_DOUBLE,             /* send data type */
                    x,                      /* recv buffer */
                    sendcounts,             /* recv counts */
                    displs,
                    MPI_DOUBLE,             /* recv data type */
                    MPI_COMM_WORLD);        /* communicator */

    /* y = alpha * P_link * x_k */
    csr_multiply(row_end-row_start, value_p, colind_p, rbegin_p, x, &x_p);
    p_sum = 0.0f;
    for (i=0;i<row_end-row_start;i++){
      x_p[i] = ALPHA * x_p[i];
      p_sum += x_p[i];
    }

    /* Reduce sum of y, and then update by adding random surfer term */
    MPI_Allreduce(&p_sum,&total_sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    total_sum = (1 - total_sum) / dimension;
    for (i=0;i<row_end-row_start;i++)
      x_p[i] = x_p[i] + total_sum;

    /* Assemble the resulting vector */
    MPI_Gatherv(  x_p,                    /* send buffer */
                  row_end-row_start,      /* send counts */
                  MPI_DOUBLE,             /* send data type */
                  y,                      /* recv buffer */
                  sendcounts,             /* send counts */
                  displs,                 /* recv counts */
                  MPI_DOUBLE,             /* recv data type */
                  0,                      /* root */
                  MPI_COMM_WORLD);        /* communicator */

    /* Leave if converge */
    if (!world_rank){
      if (sup_norm(dimension,x,y) < ERROR)
        stop_flag = 1;
    }

    /* Notify if we are finished */
    MPI_Bcast(&stop_flag,1,MPI_INT,0,MPI_COMM_WORLD);
    if (stop_flag)
      break;

    iter += 1;
  } while (1);

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
      fprintf(fd,"%lf\n",y[i]);
    }
    fclose(fd);
  }


  /* free memory */
  free(x_p);
  free(x);
  free(sendcounts);
  free(displs);
  free(value_p);
  free(colind_p);
  free(rbegin_p);
  /* end MPI environment */
  MPI_Finalize();

  return 0;
}

void readfile(char *fileName, int *n, int *m, double **value, int **colind, int **rbegin){
	FILE *fp=fopen(fileName,"r");
	char buf[200];
	int i,j,k,l;
	int p,q;
	double w;
	if ((fp=fopen(fileName,"r"))==NULL){
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
	(*value)=(double*)malloc(sizeof(double)*(*m));
	(*colind)=(int*)malloc(sizeof(int)*(*m));
	(*rbegin)=(int*)malloc(sizeof(int)*((*n)+1));
	k=-1;
	for(i=0;i<(*m);i++){
	  fgets(buf,200,fp);
	  l=0;p=atoi(&buf[l]);
	  while(buf[l++]!=' '); q=atoi(&buf[l]);
	  while(buf[l++]!=' '); w=atof(&buf[l]);

	  (*value)[i]=w;
	  (*colind)[i]=q;
	  if(p!=k){
	    k=p;
	    (*rbegin)[p]=i;
	  }
	}
	(*rbegin)[*n]=(*m);
	fclose(fp);
}


void output(int n, double *answer){
	FILE *fp=fopen("output.txt","w");
	int i;
	for(i=0;i<n;i++){
	  fprintf(fp,"%.16f\n",answer[i]);
	}
	fclose(fp);
}

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

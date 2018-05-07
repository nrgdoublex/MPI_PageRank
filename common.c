#include "common.h"

/*
 *  Summary:
 *      Initialize Options data structure
 *
 *  Output Parameters:
 *      options: Options data structure
 *
 */
void init_options(Options **options)
{
  *options = (Options *)malloc(sizeof(Options));
  (*options)->input_name = (*options)->output_name = NULL;
  (*options)->time_limit = -1.0f;
  (*options)->max_iter = -1;
  (*options)->verbose = 0;
  return;
}

/*
 *  Summary:
 *      Read options from command line arguments
 *
 *  Input Parameters:
 *      argc:  number of command line arguments
 *      argv:  array of command line arguments
 *
 *  Output Parameters:
 *      options:  Options data structure
 *
 *  Return:
 *      0 if success, 1 otherwise
 *
 */
int read_options(int argc, char **argv, Options *options)
{
  char c;

  while ((c=getopt(argc,argv,"i:o:t:m:v")) != -1){
    switch (c)
    {
      case 'i':
        if (!optarg || optarg[0] == '-'){
          fprintf(stderr,"Please identify the file name with [-i] flag.\n");
          return -1;
        }
        options->input_name = optarg;
        break;
      case 'm':
        if (optarg[0] == '-'){
          fprintf(stderr,"Please provide additional argument for [-m] flag.\n");
          return -1;
        }

        options->max_iter = atoi(optarg);
        if ( !options->max_iter){
          fprintf(stderr,"Please identify the maximum number of iterations with [-m] flag,");
          fprintf(stderr," or ignore the flag if no limit of iterations is set.\n");
          options->max_iter = -1;
        }
        break;
      case 'o':
        if (optarg[0] == '-'){
          fprintf(stderr,"Please provide additional argument for [-o] flag.\n");
          return -1;
        }

        options->output_name = optarg;
        break;
      case 't':
        if (optarg[0] == '-'){
          fprintf(stderr,"Please provide additional argument for [-t] flag.\n");
          return -1;
        }

        options->time_limit = atof(optarg);
        if (!options->time_limit){
          fprintf(stderr,"Please identify the time limit with [-t] flag,");
          fprintf(stderr," or ignore the flag if no time limit is set.\n");
          options->time_limit = -1;
        }
        break;
      case 'v':
        options->verbose = 1;
        break;
      case '?':
        if (optopt == 'i' || optopt == 'o')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,"Unknown option character `\\x%x'.\n",optopt);
        return -1;
      default:
        abort();
    }
  }

  return 0;
}

/*
 *  Summary:
 *      Read Sparse Matrix from file
 *
 *  Input Parameters:
 *      file_name:  file name of dataset
 *      verbose:  flag of further messages about the matrix
 *
 *  Output Parameters:
 *      n:      dimension of matrix
 *      m:      number of nonzero entries of matrix
 *      value:  array of nonzero values of sparse matrix
 *      colind: array of column indices of nonzero values of sparse matrix
 *      rbegin: array of number of nonzero values of sparse matrix before row[i]
 *
 */
void get_sparse_matrix( char *file_name, int *n, int *m, double **value,
                        int **colind, int **rbegin, int verbose){
	FILE *fp=fopen(file_name,"r");
	char buf[256];
	int i,j,k,l;
	int p,q;
	double w;
	if ((fp=fopen(file_name,"r"))==NULL){
	  fprintf(stderr,"Open file errer, check filename please.\n");
	}

  /* matrix dimension and number of nonzero entries */
	fgets(buf,200,fp);
	*n=atoi(buf);
	l=0;
	while(buf[l++]!=' ');
	*m=atoi(&buf[l]);

  if (verbose)
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

/*
 *  Summary:
 *      Read dataset and output row, column and values of entries
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
 *      verbose: flag to show more info about matrix
 *
 */
void get_matrix(char *filename, int *n, int *m,
              int **rowind, int **colind, double **values, int verbose)
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

  if (verbose)
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

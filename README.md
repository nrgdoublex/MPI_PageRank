# MPI_PageRank
This is the implementations of PageRank using MPI interface. The following alogrithms are implemented in this project:
* Rowwise block 1-D partitioning
* Cyclic 2-D partitioning

# File Layouts
* **pagerank_1D_par.c**: Implementation with rowwise block 1-D partitioning of matrix.
* **pagerank_2D_par.c**: Implementation with cyclic 2-D partitioning of matrix.
* **pagerank_2D_par_sqp.c**: Optimized version of **pagerank_2D_par.c**, when the number of processes is a square number.

# Syntax
mpirun -n [_num_processes_] **program_name** -i [_dataset_file_name_] -o [_output_file_name_] -m [_num_iteration_] -t [_num_seconds_] -v

# Options
* **-n**: Number of processes used. Required in MPI interface.
* **-i**: Input dataset file name. Required.
* **-o**: File name to store the resulting vector. If not specified, the vector will be printed out on the console.
* **-m**: Maximum number of iterations. If not specified, the program will keep running until resulting vector is converged.
* **-t**: Time limit of execution. If not specified, the program will keep running until resulting vector is converged.
* **-v**: Verbose mode flag.

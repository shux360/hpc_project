#!/bin/bash

IMAGE=$1

echo "===== SERIAL ====="
./serial_denoise_edge $IMAGE

echo "===== OPENMP ====="
for t in 1 2 4 8 16
do
  ./openmp_denoise_edge $IMAGE $t
done

echo "===== PTHREADS ====="
for t in 1 2 4 8 16
do
  ./pthread_denoise_edge $IMAGE $t
done

echo "===== MPI ====="
for p in 1 2 4 8
do
  mpirun -np $p ./mpi_denoise_edge $IMAGE
done

echo "===== HYBRID ====="
mpirun -np 2 ./hybrid_denoise_edge $IMAGE 2
mpirun -np 2 ./hybrid_denoise_edge $IMAGE 4
mpirun -np 4 ./hybrid_denoise_edge $IMAGE 2
mpirun -np 4 ./hybrid_denoise_edge $IMAGE 4
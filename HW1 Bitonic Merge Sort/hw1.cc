#include<bits/stdc++.h>
#include "mpi.h"
using namespace std;

void compSwap(float *a, int i, int j, int dir);
void bitonicMerge(float *a, int low, int cnt, int dir);
void bitonicLocal(float *a,int low, int cnt, int dir);

void bitonicGlobal(float *data, int chunksize, int rank, int size);
void compareHigh(int partner, float *data, int chunksize, int rank);
void compareLow(int partner, float *data, int chunksize, int rank);

void writeFilteredChunk(MPI_File output_file, float *chunk, unsigned int chunk_size);
void printArray(const char* prefix, float *arr, int size, int rank);

int np2(int x) {
    if (x < 1) return 1;
    int power = 2;
    while (power < x) power <<= 1;
    return power;
}

int main(int argc, char **argv) {
    int rank, size;
    char *input_filename = argv[2], *output_filename = argv[3];
    MPI_File input_file, output_file;

    // Create time stamps
    double start_time, end_time, io_start, io_end, comm_start, comm_end;
    double total_time = 0.0, io_time = 0.0, comm_time = 0.0;

    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    start_time = MPI_Wtime();

    if(size & (size - 1)){
        if(rank == 0) printf("Please use a process count that is a power of 2.\n");
        MPI_Finalize(); 
        return 0; 
    } 

    // Data distribution and read input file
    int arrSize, chunksize;
    unsigned int paddedSize;
    float *arr = NULL, *chunk;
	
	if (rank == 0){
        arrSize = atoi(argv[1]);
        if (!(arrSize && !(arrSize & (arrSize - 1)))) 
            paddedSize = np2(arrSize);
        else
            paddedSize = arrSize;

        arr = (float *)malloc(paddedSize * sizeof(float));
        chunksize = paddedSize / size;
    }

    comm_start = MPI_Wtime();
    MPI_Bcast(&chunksize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&paddedSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_time += (comm_end - comm_start);

    int display = (paddedSize / size) * rank;
    chunk = (float *)malloc(chunksize * sizeof(float));

    io_start = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_Status status;
    MPI_File_read_at(input_file, sizeof(float) * display, chunk, chunksize, MPI_FLOAT, &status);
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    io_end = MPI_Wtime();
    io_time += (io_end - io_start);

    int element;
    MPI_Get_count(&status, MPI_FLOAT, &element);
    for (int i = element; i < chunksize; i++){
        chunk[i] = FLT_MAX;
    }

    io_start = MPI_Wtime();
    MPI_File_close(&input_file);
    io_end = MPI_Wtime();
    io_time += (io_end - io_start);

    // Local sort
	bitonicLocal(chunk, 0, chunksize, 1);

    // Global sort
    comm_start = MPI_Wtime();
    bitonicGlobal(chunk, chunksize, rank, size);

    // Gather sorted result and write to file
    MPI_Gather(chunk, chunksize, MPI_FLOAT, arr, chunksize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_time += (comm_end - comm_start);
    
    io_start = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (rank == 0)writeFilteredChunk(output_file, arr, paddedSize);
    MPI_File_close(&output_file);
    io_end = MPI_Wtime();
    io_time += (io_end - io_start);

    end_time = MPI_Wtime();
    total_time = end_time - start_time;
    double comp_time = total_time - (io_time + comm_time);
    if (rank == 0) {
        printf("Total Time: %f\n", end_time - start_time);
        printf("I/O Time: %f\n", io_time);
        printf("Communication Time: %f\n", comm_time);
        printf("Computing Time: %f\n", comp_time);
    }

    free(chunk);
    MPI_Finalize();
    return 0;
}

// Local bitonic sort functions
void compSwap(float *a, int i, int j, int dir)
{
    if (dir==(a[i]>a[j]))
        swap(a[i], a[j]);
}
 
void bitonicMerge(float *a, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compSwap(a, i, i + k, dir);
        }
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low + k, k, dir);
    }
}

 
void bitonicLocal(float *a, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicLocal(a, low, k, 1); 
        bitonicLocal(a, low + k, k, 0); 
        bitonicMerge(a, low, cnt, dir); 
    }
}

 
// Global bitonic sort functions
void bitonicGlobal(float *data, int chunksize, int rank, int size) {
    //cout << "process " << rank << ", chunksize=" << chunksize << endl;
    int d = log2(size);
    for (int i = 0; i < d; ++i) {
        for (int j = i; j >= 0; --j) {
            int partner = rank ^ (1 << j);
            bool ascending = ((rank >> (i + 1)) % 2 == 0);
            
            if (((rank >> j) & 1) == 0) {
                if (ascending) {
                    compareLow(partner, data, chunksize, rank);
                } else {
                    compareHigh(partner, data, chunksize, rank);
                }
            } else {
                if (ascending) {
                    compareHigh(partner, data, chunksize, rank);
                } else {
                    compareLow(partner, data, chunksize, rank);
                }
            }
        }
    }
}

void compareLow(int partner, float *data, int chunksize, int rank) {

    float* receivedData = (float*)malloc(chunksize * sizeof(float));
    MPI_Sendrecv(data, chunksize, MPI_FLOAT, partner, 0,
                 receivedData, chunksize, MPI_FLOAT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    float* mergedData = (float*)malloc((chunksize*2) * sizeof(float));
    int i = 0, j = 0, k = 0;
    while (i < chunksize && j < chunksize) {
        if (data[i] < receivedData[j]) {
            mergedData[k++] = data[i++];
        } else {
            mergedData[k++] = receivedData[j++];
        }
    }
    // Copy remaining elements
    while (i < chunksize) mergedData[k++] = data[i++];
    while (j < chunksize) mergedData[k++] = receivedData[j++];

    memcpy(data, mergedData, chunksize * sizeof(float));
    
    free(receivedData);
    free(mergedData);
}


void compareHigh(int partner, float *data, int chunksize, int rank) {

    float* receivedData = (float*)malloc(chunksize * sizeof(float));
    MPI_Sendrecv(data, chunksize, MPI_FLOAT, partner, 0,
                 receivedData, chunksize, MPI_FLOAT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Reverse merge
    float* mergedData = (float*)malloc((chunksize*2) * sizeof(float));
    int i = chunksize - 1, j = chunksize - 1, k = chunksize*2 - 1;
    while (i >= 0 && j >= 0) {
        if (data[i] > receivedData[j]) {
            mergedData[k--] = data[i--];
        } else {
            mergedData[k--] = receivedData[j--];
        }
    }
    while (i >= 0) mergedData[k--] = data[i--];
    while (j >= 0) mergedData[k--] = receivedData[j--];

    memcpy(data, &mergedData[chunksize], chunksize * sizeof(float));
    
    free(receivedData);
    free(mergedData);
}

// Filter the paddings and write to file
void writeFilteredChunk(MPI_File output_file, float *chunk, unsigned int chunksize){
    float *filtered_chunk = (float *)malloc(sizeof(float) * chunksize);
    int filtered_size = 0;

    for (int i = 0; i < chunksize; ++i){
        if (chunk[i] == FLT_MAX)
            break;
        filtered_chunk[filtered_size++] = chunk[i];
    }
    MPI_File_write_at(output_file, 0, filtered_chunk, filtered_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    free(filtered_chunk);
}

// Print array to debug
void printArray(const char* prefix, float *arr, int size, int rank) {
    printf("%s - Process %d: [", prefix, rank);
    for (int i = 0; i < size; i++) {
        printf("%.1f", arr[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}
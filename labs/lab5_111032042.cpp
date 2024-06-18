#include <bits/stdc++.h>
#include <openacc.h>
#include "mnist/mnist_reader.hpp"
using namespace std;

#define RESULT_FILE "result.txt"
#define MNIST_DATA_LOCATION "/home/pp23/share/lab5/testcases/MNIST"
#define WEIGHT_ROOT "/home/pp23/share/lab5/testcases/weights"

/* Model architecture:
 *  Layer1: 784 x 1024
 *  Layer2: 1024 x 10
 */
#define LAYER0 784
#define LAYER1 1024
#define LAYER2 10

/* https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
 * D := AxB + C
 *
 * A := n*k matrix
 * B := k*m matrix
 * C := m*1 matrix
 */
 
 /* TODO: Parallel the for loops
  * HINT: 1. (a) copy array A, B, C to GPU device
  *          (b) copy array D back to CPU
  *       2. Parallel the loop using
  *          (a) #pragma acc XXX
  *          (b) CUDA kernel function
  */

__global__ void LinearLayerKernel(float *A, float *B, float *C, float *D, int n, int k, int m) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // Set tile size 32*32
    __shared__ float sA[32][32]; 
    __shared__ float sB[32][32]; 

    float sum = 0.0;
    for (int p = 0; p < (k + 31) / 32; ++p) {
        if (row < n && p * 32 + tx < k) sA[ty][tx] = A[row * k + p * 32 + tx];
        else sA[ty][tx] = 0.0;
        
        if (col < m && p * 32 + ty < k) sB[ty][tx] = B[(p * 32 + ty) * m + col];
        else sB[ty][tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < 32; ++i) sum += sA[ty][i] * sB[i][tx];
        
        __syncthreads();
    }
    if (row < n && col < m) D[row * m + col] = sum + C[col];
}


void LinearLayer(float *A, float *B, float *C, float *D, int n, int k, int m) {
    float *d_A, *d_B, *d_C, *d_D;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_A, n * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * m * sizeof(float));
    cudaMalloc((void **)&d_C, m * sizeof(float));
    cudaMalloc((void **)&d_D, n * m * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * sizeof(float), cudaMemcpyHostToDevice);

    // Define number of threads and blocks
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    LinearLayerKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_D, n, k, m);

    // Copy result back to CPU
    cudaMemcpy(D, d_D, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
}


/* https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
 * A := sigmoid(A)
 * A := n*m matrix
 */

/* TODO: Parallel the for loops */

__global__ void SigmoidKernel(float *A, int n, int m) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m) {
        int index = i * m + j;
        float value = A[index];
        A[index] = 1.0f / (1.0f + __expf(-value)); 
    }
}


void Sigmoid(float *A, int n, int m) {
    float *d_A;
    cudaMalloc((void **)&d_A, n * m * sizeof(float));
    cudaMemcpy(d_A, A, n * m * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    SigmoidKernel<<<numBlocks, threadsPerBlock>>>(d_A, n, m);

    cudaMemcpy(A, d_A, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}


/* Argmax: Choose the index with the largest value
 * A := n*m matrix (data type: float)
 * D := n*1 matrix (data type: int)
 */

 /* TODO: Parallel the for loops */

__global__ void ArgmaxKernel(float *A, int *D, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float mx = A[i * m];
        int index = 0;
        for (int j = 1; j < m; j++) {
            if (mx < A[i * m + j]) {
                mx = A[i * m + j];
                index = j;
            }
        }
        D[i] = index;
    }
}

void Argmax(float *A, int *D, int n, int m) {
    float *d_A;
    int *d_D;
    cudaMalloc((void **)&d_A, n * m * sizeof(float));
    cudaMalloc((void **)&d_D, n * sizeof(int));
    cudaMemcpy(d_A, A, n * m * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    ArgmaxKernel<<<numBlocks, blockSize>>>(d_A, d_D, n, m);

    cudaMemcpy(D, d_D, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_D);
}


/* my_nn: A simple neural network
 * Input arrays:
 *     training_images_flat: float[num_images * LAYER0]
 *     layer1_matrix:        float[LAYER0 * LAYER1]
 *     layer1_bias:          float[LAYER1]
 *     layer2_matrix:        float[LAYER1 * LAYER2]
 *     layer2_bias:          float[LAYER2]
 * Output array:
 *     result:               int[num_images]
 */
 
void my_nn(float *training_images_flat, int num_images,
           float *layer1_matrix, float *layer1_bias, float *layer2_matrix, float *layer2_bias,
           int *result) {
    float *layer1_output = new float[num_images * LAYER1];
    float *layer2_output = new float[num_images * LAYER2];

    // Layer1: Linear layer + Sigmoid (activation function)
    LinearLayer(training_images_flat, layer1_matrix, layer1_bias, layer1_output,
                num_images, LAYER0, LAYER1);
    Sigmoid(layer1_output, num_images, LAYER1);


    // Layer2: Linear layer + Argmax
    LinearLayer(layer1_output, layer2_matrix, layer2_bias, layer2_output,
        num_images, LAYER1, LAYER2);
    Argmax(layer2_output, result, num_images, LAYER2);

    delete [] layer1_output;
    delete [] layer2_output;
}


/////////////////////////////////////////////////////////////////////
//                 NO NOT MODIFY THE CODE BELOW                    //
/////////////////////////////////////////////////////////////////////

/* Read neural network's weight from file (in binary format)
 */
void read_weight(float *array, string filename, int num_floats) {
    string full_filename = string(WEIGHT_ROOT) + '/' + filename;
    std::cout << "Reading file: " << full_filename << std::endl;
    ifstream file(full_filename, ios::in | ios::binary);
    if (!file) {
        std::cerr << "error reading file: " << full_filename << std::endl;
        exit(1);
    }
    file.read((char*)array, num_floats * sizeof(float));
}

/* Write predicted result to file
 */
void write_predict(int *result, int n, string filename) {
    std::ofstream file(filename, std::ofstream::out);
    for (int i = 0; i < n; i++) {
        file << result[i] << '\n';
    }
    file.close();
}

/* Print an image
 * Usage: print_img(training_images[i])
 */
void print_img(float *img) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (img[i*28+j] > 0.5) {
                std::cout << 'x';
            }else {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

bool InitCUDA(void) {
    int count = 0;

    count =  acc_get_num_devices(acc_device_nvidia);
    if(count == 0) {
        std::cerr << "There is no device.\n";
        return false;
    }

    acc_set_device_num(0, acc_device_nvidia);

    std::cout << "CUDA initialized.\n";
    return true;
}

int main(int argc, char* argv[]) {
    auto initcuda_start = std::chrono::steady_clock::now();
    InitCUDA();
    auto read_start = std::chrono::steady_clock::now();
    // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    /* Load MNIST data
     */
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
    if (argc == 1) {
        dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    }else {
        dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(argv[1]);
    }

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    // std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    // std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    // std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    int num_train_images = dataset.training_images.size();
    // int num_train_images = 8000;
    int num_pixels       = dataset.training_images.front().size();  // should be 28*28 = 784

    /* Convert 60000 training images from [0, 255] to [0, 1)
     * We will first generate another 2D array by `new`
     */

    /* training_images_flat[i*num_pixels + j] == training_images[i][j]
     * j-th pixel in i-th image
     */
    float *training_images_flat = new float[num_train_images * num_pixels];

    float **training_images = new float*[num_train_images];
    for (int i = 0; i < num_train_images; i++) {
        training_images[i] = training_images_flat + i * num_pixels;
    }

    for (int i = 0; i < num_train_images; i++) {
        for (int j = 0; j < num_pixels; j++) {
            training_images[i][j] = (float)(dataset.training_images[i][j]) / 255.0;
        }
    }

    /* Print first image */
    // print_img(training_images[0]);
    

    /* Load matrices' weight from binary file
     * You can print the binary file by: `od -f layer1_bais`
     * https://stackoverflow.com/questions/36791622/how-to-print-float-value-from-binary-file-in-shell
     */
    float *layer1_matrix = new float[LAYER0 * LAYER1];
    float *layer1_bias = new float[LAYER1];
    float *layer2_matrix = new float[LAYER1 * LAYER2];
    float *layer2_bias = new float[LAYER2];
    read_weight(layer1_matrix, "layer1_matrix", LAYER0 * LAYER1);
    read_weight(layer1_bias, "layer1_bias", LAYER1);
    read_weight(layer2_matrix, "layer2_matrix", LAYER1 * LAYER2);
    read_weight(layer2_bias, "layer2_bias", LAYER2);
    
    
    /*
    std::cout << "The first 10 numbers in layer1_matrix: ";
    for (int i = 0; i < 10; i++) {
        std::cout << layer1_matrix[i] << ' ';
    }
    std::cout << std::endl;
    */

    auto read_end = std::chrono::steady_clock::now();

    /* Inference */
    int *result = new int[num_train_images];
    my_nn(training_images_flat, num_train_images,
          layer1_matrix, layer1_bias, layer2_matrix, layer2_bias, result);

    auto inference_end = std::chrono::steady_clock::now();

    /* Calculate accuracy */
    int correct = 0;
    int total = 0;
    for (int i = 0; i < num_train_images; i++) {
        if ((int)result[i] == (int)dataset.training_labels[i]) {
            correct++;
        }
        total++;
    }
    std::cout << "\nInference accuracy: " << (double)correct / (double)total * 100.0 << "%\n";
    if (argc == 1) 
        write_predict(result, num_train_images, RESULT_FILE);
    else
        write_predict(result, num_train_images, argv[2]);

    auto acc_end = std::chrono::steady_clock::now();

    std::cout << std::setprecision(5) << std::fixed;
    std::cout << "\n-----     STATS     -----\n";
    std::cout << "Time for initializing CUDA device:     " << std::chrono::duration_cast<std::chrono::milliseconds>(read_start - initcuda_start).count() << " m.s.\n";
    std::cout << "Time for reading MNIST data & weights: " << std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start).count() << " m.s.\n";
    std::cout << "Time for inferencing                 : " << std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - read_end).count() << " m.s.\n";
    std::cout << "Time for calculating accuracy        : " << std::chrono::duration_cast<std::chrono::milliseconds>(acc_end - inference_end).count() << " m.s.\n";
    std::cout <<   "----- END OF STATS  -----\n";

    delete [] result;
    delete [] layer1_matrix;
    delete [] layer1_bias;
    delete [] layer2_matrix;
    delete [] layer2_bias;
    delete [] training_images_flat;
    delete [] training_images;
    return 0;
}

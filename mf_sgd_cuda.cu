#include<time.h>
#include<stdlib.h>
#include<iostream>
#include<vector>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<sstream>
#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cmath>
#include<time.h>
#include<tuple>
#include<limits>
#include<getopt.h>
#include<fstream>
#include<random>
#include<math.h>
#include<curand.h>
#include<curand_kernel.h>
#include<stdio.h>
#include<stdexcept>
#include<tuple>
#define index(i, j, N)  ((i)*(N)) + (j)
enum ErrorType { MAE, RMSE };
using namespace std;


// hyper paramters
//cached in the device
__constant__ int cur_iterations_d = 0;
__constant__ int total_iterations_d = 1000;
__constant__ int n_factors_d = 10;
__constant__ float learning_rate_d = 1e-4;
__constant__ int seed_d = 42;
__constant__ float P_reg_d = 1e-1;
__constant__ float Q_reg_d = 1e-1;
__constant__ float user_bias_reg_d = 1e-1;
__constant__ float item_bias_reg_d = 1e-1;
__constant__ bool is_train_d = true;


__global__ void initCurand(curandState *state, unsigned long seed, int n_rows){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < n_rows) {
        curand_init(seed, x, 0, &state[x]);
    }
}
// being called in the sgd update and the loss kernels
__device__ float get_prediction(int factors, const float *p, const float *q, float user_bias, float item_bias, float global_bias) {
        float pred = global_bias + user_bias + item_bias;
        for (int f = 0; f < factors; f++)
            pred += q[f]*p[f];
        return pred;
}

__global__ void sgd_update(int *indptr, int *indices, const float *data, float *P, float *Q, float *Q_target, 
                           int n_rows, float *user_bias, float *item_bias,
                           float *item_bias_target, curandState *my_curandstate,
                           float global_bias, int start_user, bool *item_is_updated) {
    // One thread per user
    int x = (blockDim.x * blockIdx.x + threadIdx.x + start_user) % n_rows;
    if(x < n_rows) {

        // pick a random item y_i
        // to check for the count of the item ratings
        int low = indptr[x];
        int high = indptr[x+1];

        // Does the user have any rating at all? If he has, proceed to the if loop
        if(low != high) {

            float myrandf = curand_uniform(&my_curandstate[x]); // random double between (0, 1]
            int y_i = (int) ceil(myrandf * (high - low)) - 1 + low; // random integer between [low, high)

            // selecting a random item to update on
            // move some reused values to registers
            int y = indices[y_i];
            float ub = user_bias[x];
            float ib = item_bias[y];

            // get the error random item y_i
            //error = realrating - predictedrating
            // x is the thread id and n_factors is the hyperparameter
            float error_y_i = data[y_i] - get_prediction(n_factors_d, &P[x * n_factors_d], &Q[y * n_factors_d], ub, ib, global_bias);

            // do not update on the item which is already updated by other thread, to reduce excessive access
            bool updated = !item_is_updated[y];
            item_is_updated[y] = true;

            // update components
            for(int f = 0; f < n_factors_d; ++f) {
                //index(i, j, N)  ((i)*(N)) + (j)
                float P_old = P[index(x, f, n_factors_d)];
                float Q_old = Q[index(y, f, n_factors_d)];

                // update components
                P[index(x, f, n_factors_d)] = P_old + learning_rate_d * (error_y_i * Q_old - P_reg_d * P_old);

                // Only update Q if train flag is true and thread is the early bird
                // update the Q component for training phase 
                if(is_train_d && updated) {
                    Q_target[index(y, f, n_factors_d)] = Q_old+ learning_rate_d * (error_y_i * P_old - Q_reg_d * Q_old);
                }
            }

            // update user bias
            user_bias[x] += learning_rate_d * (error_y_i - user_bias_reg_d * ub);

            // Only update item_bias if train flag is true and thread is the early bird
            if(is_train_d && updated) {
                item_bias_target[y] = ib + learning_rate_d * (error_y_i - item_bias_reg_d * ib);
            }
        }
    }
}


// for the array of structure on the host device
struct Rating
{
    int userID;
    int itemID;
    float rating;
};
class Config {
    public:        
        int cur_iterations = 0;         // Current iteration count
        int total_iterations = 5000;    // Total iteration count
        int n_factors = 50;             // Number of latent factors to use
        float learning_rate = 0.01;     // The learning rate for SGD
        int seed = 42;                  // The seed for the random number generator
        float P_reg = 0.02;             // The regularization parameter for the user matrix
        float Q_reg = 0.02;             // The regularization parameter for the item matrix
        float user_bias_reg = 0.02;     // The regularization parameter for user biases
        float item_bias_reg = 0.02;     // The regularization parameter for item biases
        bool is_train = true;           // Whether we're doing full training or partial fit        
        int n_threads = 32;             // The number of threads in a block    
        int check_error = 500;          // The number of iterations before calculating loss 
        float patience = 2;             // The number of times loss can stay constant or increase before triggering a learning rate decay
        float learning_rate_decay = 0.2;// The amount of decay for learning rate. When patience


    // Read configs from the file
    bool read_config(std::string file_path) {
        std::ifstream config_file(file_path);
        config_file >> cur_iterations >> total_iterations >> n_factors >> learning_rate >>
            seed >> P_reg >> Q_reg >> user_bias_reg >> item_bias_reg;
        config_file.close();
        return true;
    }

    // not necessary, but you can write back and configurations
    bool write_config(std::string file_path) {
        std::ofstream config_file(file_path);
        config_file << cur_iterations << " " << total_iterations << " " << n_factors << " " <<
        learning_rate << " " << seed << " " << P_reg << " " << Q_reg << " " << user_bias_reg <<
        " " << item_bias_reg << "\n";
        config_file.close();
        return true;
    }

    // now copy the hyperparameters onto device's constant memory
    bool set_cuda_variables() {
        cudaMemcpyToSymbol(cur_iterations_d, &cur_iterations, sizeof(int));
        cudaMemcpyToSymbol(total_iterations_d, &total_iterations, sizeof(int));
        cudaMemcpyToSymbol(n_factors_d, &n_factors, sizeof(int));
        cudaMemcpyToSymbol(learning_rate_d, &learning_rate, sizeof(float));
        cudaMemcpyToSymbol(seed_d, &seed, sizeof(int));
        cudaMemcpyToSymbol(P_reg_d, &P_reg, sizeof(float));
        cudaMemcpyToSymbol(Q_reg_d, &Q_reg, sizeof(float));
        cudaMemcpyToSymbol(user_bias_reg_d, &user_bias_reg, sizeof(float));
        cudaMemcpyToSymbol(item_bias_reg_d, &item_bias_reg, sizeof(float));
        return true;
    }
    // not necessary to do this
    // now copy the hyperparameters onto device's constant memory
    bool get_cuda_variables() {
        cudaMemcpyFromSymbol(&cur_iterations, cur_iterations_d, sizeof(int));
        cudaMemcpyFromSymbol(&total_iterations, total_iterations_d, sizeof(int));
        cudaMemcpyFromSymbol(&n_factors,n_factors_d, sizeof(int));
        cudaMemcpyFromSymbol(&learning_rate, learning_rate_d, sizeof(float));
        cudaMemcpyFromSymbol(&seed, seed_d, sizeof(int));
        cudaMemcpyFromSymbol(&P_reg, P_reg_d, sizeof(float));
        cudaMemcpyFromSymbol(&Q_reg, Q_reg_d, sizeof(float));
        cudaMemcpyFromSymbol(&user_bias_reg, user_bias_reg_d, sizeof(float));
        cudaMemcpyFromSymbol(&item_bias_reg, item_bias_reg_d, sizeof(float));
        return true;
    }

    void print_config() {
        printf("Hyperparameters:\n");
        printf("total_iterations: %d\n", total_iterations);
        printf("n_factors: %d\n", n_factors);
        printf("learning_rate: %f\n", learning_rate);
        printf("P_reg: %f\n", P_reg);
        printf("Q_reg: %f\n", Q_reg);
        printf("user_bias_reg: %f\n", user_bias_reg);
        printf("item_bias_reg: %f\n", item_bias_reg);
        printf("is_train: %s\n", is_train?"true":"false");
        printf("n_threads: %d\n", n_threads);
        printf("check_error: %d\n", check_error);
        printf("patience: %f\n", patience);
        printf("learning_rate_decay: %f\n", learning_rate_decay);
    }        
};


//Array of Structs
vector<Rating> read_file(string filename, int *rows, int *cols, float *global_bias) {
    // read in the csv file into array of struct
    int max_row = 0;
    int max_col = 0;
    double sum_ratings = 0;
    std::ifstream ratingsFile(filename);


    // ARRAY of STRUCTURES
    std::vector<Rating> ratings;

    //cout << "in read file"  <<filename << endl;
    if (ratingsFile.is_open()){
        int userID, itemID;
        float rating;
        char delimiter;        
        ratingsFile.ignore(1000, '\n');
        while(ratingsFile >> userID >> delimiter >> itemID >> delimiter >> rating) {
            ratings.push_back({userID - 1, itemID - 1, rating});
            max_row = std::max(userID, max_row);
            max_col = std::max(itemID, max_col);
            sum_ratings += rating;
        }
        *rows = max_row;
        *cols = max_col;
        *global_bias = sum_ratings / (1.0 * ratings.size());
        return ratings;
    }
    else{
        std::cerr<<"ERROR: The file isnt open.\n";
        return ratings;
    }
}
void printRating(Rating r){
    std::cout << r.userID << "  "<< r.itemID <<"  "<< r.rating << "\n";
}

void printCSV(std::vector<Rating> *ratings) {
    // Print the vector
    std::cout  << "UserID" << "   ItemID" << "   Rating\n";
    for (int x(0); x < ratings->size(); ++x){
        printRating(ratings->at(x));
    }
}


//Structure of Arrays
class CudaCSRMatrix {
        public:
        int * indptr, * indices;
        float * data;
        int rows, cols, nonzeros;
        CudaCSRMatrix(int rows, int cols, int nonzeros,const int * indptr_, const int * indices_, const float * data_){
            this->rows = rows;
            this -> cols = cols;
            this -> nonzeros = nonzeros;
            cudaMalloc(&indptr, (rows + 1) * sizeof(int));
            cudaMemcpy(indptr, indptr_, (rows + 1)*sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc(&indices, nonzeros * sizeof(int));
            cudaMemcpy(indices, indices_, nonzeros * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc(&data, nonzeros * sizeof(float));
            cudaMemcpy(data, data_, nonzeros * sizeof(int), cudaMemcpyHostToDevice);            
        }
        ~CudaCSRMatrix() {
            cudaFree(indices);
            cudaFree(indptr);
            cudaFree(data);
        }   

};

class CudaDenseMatrix {
        public:
        int rows, cols;
        float * data;
        // for storing the dense matrices such as the P and Q components on the GPU without CSR representation
        CudaDenseMatrix(int rows, int cols, const float * host_data){
            this->rows = rows;
            this -> cols = cols;
            cudaMalloc(&data, rows * cols * sizeof(float));
            if (host_data) {
            cudaMemcpy(data, host_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
            }
        }
        ~CudaDenseMatrix(){
            cudaFree(data);
        }
        void to_host(float * output) const{
            cudaMemcpy(output, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
        }
};

float* initialize_normal_array(int size, int n_factors, float mean, float stddev, int seed) {
    mt19937 generator(seed);
    std::normal_distribution<float> distribution(mean, stddev / n_factors);
    float *array = new float[size];
    for(int i = 0; i < size; ++i) {
        array[i] = distribution(generator);
    }
    return array;
}

float* initialize_normal_array(int size, int n_factors, float mean, float stddev) {
    return initialize_normal_array(size, n_factors, mean, stddev, 42);
}

float* initialize_normal_array(int size, int n_factors, int seed) {
    return initialize_normal_array(size, n_factors, 0, 1, seed);
}

float *initialize_normal_array(int size, int n_factors) {
    return initialize_normal_array(size, n_factors, 0, 1);
}
CudaCSRMatrix* convertToCSR(std::vector<Rating> *ratings, int rows, int cols) {

    std::vector<int> indptr_vec;
    int *indices = new int[ratings->size()];
    float *data = new float[ratings->size()];
    int lastUser = -1;
    for(int i = 0; i < ratings->size(); ++i) {
        Rating r = ratings->at(i);
        if(r.userID != lastUser) {
            while(lastUser != r.userID) {
                indptr_vec.push_back(i);
                lastUser++;
            }
        }
        indices[i] = r.itemID;
        data[i] = r.rating;
    }
    indptr_vec.push_back(ratings->size());
    int *indptr = indptr_vec.data();

    // Create the Sparse Matrix
    const int *indptr_c = const_cast<const int*>(indptr);
    const int *indices_c = const_cast<const int*>(indices);
    const float *data_c = const_cast<const float*>(data);
    CudaCSRMatrix* matrix = new CudaCSRMatrix(rows, cols, (int)(ratings->size()), indptr_c, indices_c, data_c);
    cudaDeviceSynchronize();

    return matrix;
}

__global__ void loss_kernel(int factors, int user_count, int item_count, const float * P, const float * Q, const int * indptr, 
                            const int * indices, const float * data, float * error, float * user_bias, float * item_bias, float global_bias) {
    
    // One thread per user
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if(u < user_count) {
        // Get this user's factors and bias
        const float * p = &P[u * factors];
        const float ub = user_bias[u];

        // Loop over all items of user
        for (int i = indptr[u]; i < indptr[u + 1]; ++i) {
            int item_id = indices[i];
            error[i] = data[i] - get_prediction(factors, p, &Q[item_id * factors], ub, item_bias[item_id], global_bias);
        }
    }
}

/** Driver function for calculating the losses per item. Puts the losses in the error_d array.*/
void calculate_loss_gpu(CudaDenseMatrix* P_d, CudaDenseMatrix* Q_d, Config* cfg, int user_count, int item_count, int num_ratings, 
                        CudaCSRMatrix* matrix, float * error_d, float * user_bias,  float * item_bias, float global_bias) {
    dim3 dimBlock(cfg->n_threads);
    dim3 dimGrid(user_count / cfg->n_threads + 1);
    loss_kernel<<<dimGrid, dimBlock>>>(
        cfg->n_factors, user_count, item_count, P_d->data, Q_d->data,
        matrix->indptr, matrix->indices, matrix->data, error_d,
        user_bias, item_bias, global_bias);
    cudaGetLastError();
}

template <unsigned int block_size>
__global__ void total_loss_kernel(float *in_errors, double *out_errors, int n_errors, ErrorType error_type) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * block_size + tid;
    unsigned int grid_size = block_size * gridDim.x;
    sdata[tid] = 0;
    // Each thread does n_errors / grid_size work to start, before reduction
    while (i < n_errors) {
        // Each error is actual_rating - predicted_rating.
        // We want its square for RMSE and its absolute value for MAE
        sdata[tid] += error_type == RMSE ? pow(in_errors[i], 2) : abs(in_errors[i]);
        i += grid_size;
    }
    __syncthreads();

    // Start reduction. This is an unrolled loop from 512 to 1.
    // Note that any if statements related to the block size are
    // completely skipped because of templating
    if (block_size >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            sdata[tid] += sdata[tid + 32];
        }
        __syncthreads();
    }
    // if block_size is 1, compiler will complain about unneeded unsigned
    // int comparison (tid) to a value of 0
    if (!(block_size == 1) && tid < block_size / 2) {
        if (block_size >= 32) {
            sdata[tid] += sdata[tid + 16];
            __syncthreads();
        }
        if (block_size >= 16) {
            sdata[tid] += sdata[tid + 8];
            __syncthreads();
        }
        if (block_size >= 8) {
            sdata[tid] += sdata[tid + 4];
            __syncthreads();
        }
        if (block_size >= 4) {
            sdata[tid] += sdata[tid + 2];
            __syncthreads();
        }
        if (block_size >= 2) {
            sdata[tid] += sdata[tid + 1];
            __syncthreads();
        }
    }
    // All the errors in the block are now reduced to the first
    // index of shared data
    if (tid == 0) out_errors[blockIdx.x] = sdata[0];
}

/** Calculates the total Mean Absolute Error and Root Mean Squared Error on CPU.
 */
std::tuple<float, float> get_error_metrics_cpu(float *errors, float *errors_device, int n_errors) {
    cudaMemcpy(errors, errors_device, n_errors * sizeof(float), cudaMemcpyDeviceToHost);
    double mae = 0.0;
    double rmse = 0.0;
    for(int k = 0; k <  n_errors; k++) {
        mae += abs(errors[k]);
        rmse += errors[k] * errors[k];
    }
    mae /= n_errors;
    rmse = sqrt(rmse / n_errors);
    return std::make_tuple((float)mae, (float)rmse);
}

/** Convenience function for adding up the total loss.
 * The block size must be a power of 2, up to 512.
 * Since the block_size in template needs to be known at compile time,
 * this is a workaround for making it variable.
 */
float calculate_error_metric_gpu(float *in_errors, double *out_errors, double *out_errors_host, int n_errors, int grid_size, int block_size, ErrorType error_type) {
    switch(block_size) {
        case 512:
            total_loss_kernel<512><<<grid_size, block_size, 512 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 256:
            total_loss_kernel<256><<<grid_size, block_size, 256 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 128:
            total_loss_kernel<128><<<grid_size, block_size, 128 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 64:
            total_loss_kernel< 64><<<grid_size, block_size,  64 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 32:
            total_loss_kernel< 32><<<grid_size, block_size,  32 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 16:
            total_loss_kernel< 16><<<grid_size, block_size,  16 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 8:
            total_loss_kernel<  8><<<grid_size, block_size,   8 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 4:
            total_loss_kernel<  4><<<grid_size, block_size,   4 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 2:
            total_loss_kernel<  2><<<grid_size, block_size,   2 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
        case 1:
            total_loss_kernel<  1><<<grid_size, block_size,   1 * sizeof(double)>>>(in_errors, out_errors, n_errors, error_type);
            break;
    }
    cudaGetLastError();
    // Sum up the blocks in CPU
    cudaMemcpy(out_errors_host, out_errors, grid_size * sizeof(double), cudaMemcpyDeviceToHost);
    double total = 0;
    for(int k = 0; k < grid_size; k++) {
        total += out_errors_host[k];
    }
    return error_type == RMSE ? sqrt(total / n_errors) : total / n_errors;
}

/** Calculates the total Mean Absolute Error and Root Mean Squared Error on GPU.
 * Needs to call the kernel twice, but is still faster than the CPU version.
 */
std::tuple<float, float> get_error_metrics_gpu(float *in_errors, double *out_errors, double *out_errors_host, int n_errors, int grid_size, int block_size) {
    float mae = calculate_error_metric_gpu(in_errors, out_errors, out_errors_host, n_errors, grid_size, block_size, MAE);
    float rmse = calculate_error_metric_gpu(in_errors, out_errors, out_errors_host, n_errors, grid_size, block_size, RMSE);
    return std::make_tuple(mae, rmse);
}


void train(CudaCSRMatrix* train_matrix,CudaCSRMatrix* test_matrix,Config* cfg, float **P_ptr, float **Q_ptr, float *Q, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float *item_bias, float global_bias) {
    int user_count = train_matrix->rows;
    int item_count = train_matrix->cols;
    cfg->set_cuda_variables();

    // Initialize P, Q has already been initialized
    float *P = initialize_normal_array(user_count * cfg->n_factors, cfg->n_factors);
    float *losses = new float[cfg->total_iterations];
    *P_ptr = P;
    *losses_ptr = losses;

    // Copy P and Q to device memory
    CudaDenseMatrix* P_device = new CudaDenseMatrix(user_count, cfg->n_factors, P);
    CudaDenseMatrix* Q_device = new CudaDenseMatrix(item_count, cfg->n_factors, Q);
    CudaDenseMatrix* P_device_target = new CudaDenseMatrix(user_count, cfg->n_factors, P);
    CudaDenseMatrix* Q_device_target = new CudaDenseMatrix(item_count, cfg->n_factors, Q);

    // Flag to check if item's weights have been updated for each iteration
    bool *item_is_updated;
    cudaMalloc(&item_is_updated, item_count * sizeof(bool));
    cudaMemset(item_is_updated, false, item_count * sizeof(bool));

    // Create the errors
    float *errors = new float[train_matrix->nonzeros];
    float *errors_device;
    cudaMalloc(&errors_device, train_matrix->nonzeros * sizeof(float));

    float *errors_test = new float[test_matrix->nonzeros];
    float *errors_test_device;
    cudaMalloc(&errors_test_device, test_matrix->nonzeros * sizeof(float));

    // Create the bias array
    float *user_bias = initialize_normal_array(user_count, cfg->n_factors);
    *user_bias_ptr = user_bias;
    
    float *user_bias_device;
    cudaMalloc(&user_bias_device, user_count * sizeof(float));
    cudaMemcpy(user_bias_device, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice);

    float *item_bias_device;
    cudaMalloc(&item_bias_device, item_count * sizeof(float));
    cudaMemcpy(item_bias_device, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice);

    // Create bias targets
    float *user_bias_target, *item_bias_target;
    cudaMalloc(&user_bias_target, user_count * sizeof(float));
    cudaMemcpy(user_bias_target, user_bias, user_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&item_bias_target, item_count * sizeof(float));
    cudaMemcpy(item_bias_target, item_bias, item_count * sizeof(float), cudaMemcpyHostToDevice);

    // Dimensions
    dim3 dim_block(cfg->n_threads);
    dim3 dim_grid_sgd(user_count / cfg->n_threads + 1);
    dim3 dim_grid_loss(256); 
    dim3 dim_block_loss(2 * cfg->n_threads);

    // Create loss per block
    double *block_errors_host = new double[dim_grid_loss.x];
    double *block_errors_device;
    cudaMalloc(&block_errors_device, dim_grid_loss.x * sizeof(double));
    cudaMemset(block_errors_device, 0, dim_grid_loss.x * sizeof(double));

    // Create curand state
    curandState *d_state;
    cudaMalloc(&d_state, user_count * sizeof(curandState));
    // Set up random state using iteration as seed
    initCurand<<<dim_grid_sgd, dim_block>>>(d_state, cfg->seed, user_count);
    cudaGetLastError();

    // to measure time taken by a specific part of the code 
    double time_taken;
    clock_t start, end;

    // Change the starting user in order to not bias the algorithm
    // to favor later users.
    int start_user = 0;
    int start_change_speed = 250;

    // Adaptive learning rate setup
    float train_rmse, train_mae, validation_rmse, validation_mae, last_validation_rmse;
    validation_rmse = validation_mae = std::numeric_limits<float>::max();
    int current_patience = cfg->patience;

    // Training loop
    start = clock();
    for (int i = 0; i < cfg->total_iterations; ++i) {

        // Run single iteration of SGD
        sgd_update<<<dim_grid_sgd, dim_block>>>(train_matrix->indptr, train_matrix->indices, train_matrix->data, P_device->data, Q_device->data, 
                                                Q_device_target->data, user_count, user_bias_device, item_bias_device, item_bias_target, d_state,
                                                global_bias, start_user, item_is_updated);
        cudaGetLastError();

        start_user += start_change_speed;

        // Calculate total loss first, last, and every check_error iterations
        if((i + 1) % cfg->check_error == 0 || i == 0 || (i + 1) % cfg->total_iterations == 0) {

            // Calculate error on train ratings
            calculate_loss_gpu(P_device, Q_device, cfg, user_count, item_count, train_matrix->nonzeros, train_matrix,
                               errors_device, user_bias_device, item_bias_device, global_bias);

            // Calculate error on test ratings
            calculate_loss_gpu(P_device, Q_device, cfg, test_matrix->rows, test_matrix->cols, test_matrix->nonzeros, test_matrix,
                               errors_test_device, user_bias_device, item_bias_device, global_bias);

            // save previous metrics
            last_validation_rmse = validation_rmse;

            // TODO add this as a param to train function
            bool use_gpu = true;
            if(use_gpu) {
                std::tie(train_mae, train_rmse) = get_error_metrics_gpu(errors_device, block_errors_device, block_errors_host, train_matrix->nonzeros, dim_grid_loss.x, dim_block_loss.x);
                printf("TRAIN: Iteration %d GPU MAE: %f RMSE: %f\n", i + 1, train_mae, train_rmse);
                std::tie(validation_mae, validation_rmse) = get_error_metrics_gpu(errors_test_device, block_errors_device, block_errors_host, test_matrix->nonzeros, dim_grid_loss.x, dim_block_loss.x);
                printf("TEST: Iteration %d GPU MAE: %f RMSE: %f\n", i + 1, validation_mae, validation_rmse);
            } else {
                std::tie(train_mae, train_rmse) = get_error_metrics_cpu(errors, errors_device, train_matrix->nonzeros);
                printf("TRAIN: Iteration %d MAE: %f RMSE: %f\n", i + 1, train_mae, train_rmse);
                std::tie(validation_mae, validation_rmse) = get_error_metrics_cpu(errors_test, errors_test_device, test_matrix->nonzeros);
                printf("TEST: Iteration %d MAE: %f RMSE: %f\n", i + 1, validation_mae, validation_rmse);
            }

            // Update learning rate if needed
            if(last_validation_rmse < validation_rmse) {
                current_patience--;
            }
            if(current_patience <= 0) {
                current_patience = cfg->patience;
                cfg->learning_rate *= cfg->learning_rate_decay;
                cfg->set_cuda_variables();

                printf("New Learning Rate: %f\n: ", cfg->learning_rate);
            }

            // TODO: Do we still need to store this?
            losses[i] = validation_rmse;
        }

        cudaGetLastError();

        // Swap item related components
        swap(Q_device, Q_device_target);
        swap(item_bias_device, item_bias_target);

        // Reset each item's updated status for next iteration
        cudaMemset(item_is_updated, false, item_count * sizeof(bool));

        cfg->cur_iterations += 1;
    }
    cudaDeviceSynchronize();
    end = clock();

    // Output time taken
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;   
    printf("Time taken for %d of iterations is %lf\n", cfg->total_iterations, time_taken);

    // Copy updated P and Q back
    P_device->to_host(P);
    Q_device->to_host(Q);

    // Copy updated bias arrays back
    cudaMemcpy(user_bias, user_bias_device, user_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(item_bias, item_bias_device, item_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(errors_device);
    cudaFree(errors_test_device);
    cudaFree(block_errors_device);
    cudaFree(user_bias_device);
    cudaFree(item_bias_device);
    cudaFree(user_bias_target);
    cudaFree(item_bias_target);
    cudaFree(d_state);
    cudaFree(item_is_updated);
    delete P_device;
    delete P_device_target;
    delete Q_device;
    delete Q_device_target;
    delete [] errors;
    delete [] errors_test;
    delete [] block_errors_host;
}

/** Initializes Q and item bias randomly and passes them into the training method.
 */
void train(CudaCSRMatrix* train_matrix, CudaCSRMatrix* test_matrix, Config* cfg, float **P_ptr, float **Q_ptr, float **losses_ptr,
           float **user_bias_ptr, float **item_bias_ptr, float global_bias) {
    int item_count = train_matrix->cols;
    // Initialize for regular training
    float *Q = initialize_normal_array(item_count * cfg->n_factors, cfg->n_factors);
    float *item_bias = initialize_normal_array(item_count, cfg->n_factors);
    *Q_ptr = Q;
    *item_bias_ptr = item_bias;
    train(train_matrix, test_matrix, cfg, P_ptr, Q_ptr, Q, losses_ptr, user_bias_ptr, item_bias_ptr, item_bias, global_bias);
}
void writeToFile(string parent_dir, string base_filename, string extension, string component, float *data, int rows, int cols, int factors) {
    char filename [255];
    sprintf(filename, "%s/%s_f%d_%s.%s", parent_dir.c_str(), base_filename.c_str(), factors, component.c_str(), extension.c_str());
    

    FILE *fp;
    fp = fopen(filename, "w");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols - 1; j++) {
            fprintf(fp, "%f,", data[index(i, j, cols)]);
        }
        fprintf(fp, "%f", data[index(i, cols - 1, cols)]);
        fprintf(fp, "\n");
    }
    fclose(fp);    
}

int main(int argc, char **argv){
    cout << argc;
    cout << **argv;
    if (argc <2){
        return -1;
    }
    std::string config_file;
    int option;
    
    while((option = getopt(argc, argv, "c:")) != -1) {
        switch(option) {
            case 'c':
                config_file = optarg;
                break;
            default:
                cout << "Unknown option.\n";
                return 1;
        }
    }
    size_t free_memory, total_memory;

    cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
    if (err != cudaSuccess)
    {
        cout << "getFreeBytes: call index "  << ": cudaMemGetInfo returned the error: " << cudaGetErrorString(err) << endl;
        exit(1);
    }
    cout << "Free Memory " << (long)free_memory << endl;

    string training_file = argv[optind++];
    string testing_file = argv[optind++];
    int rows, cols;
    float  global_bias;
    int r, c;
    float gb;
    //Step 1: Read the CSV file
    std::vector<Rating> train_ratings = read_file(training_file, &rows, &cols, &global_bias);
    std::vector<Rating> test_ratings = read_file(testing_file, &r, &c, &gb);
    // Step 2: Convert to CSR Representation
    CudaCSRMatrix* train_matrix = convertToCSR(&train_ratings, rows, cols);
    CudaCSRMatrix* test_matrix = convertToCSR(&test_ratings, r, c);


    
    Config *cfg = new Config();
    if(!config_file.empty())
        cfg->read_config(config_file);
    
    cfg->print_config();    

    float *P, *Q, *losses, *user_bias, *item_bias;
    train(train_matrix,test_matrix, cfg, &P, &Q, &losses, &user_bias, &item_bias, global_bias);

    size_t dir_index = training_file.find_last_of("/"); 
    string parent_dir, filename;
    if(dir_index != string::npos) {
        parent_dir = training_file.substr(0, dir_index);
        filename = training_file.substr(dir_index + 1);
    } else {
        // doesn't have directory, therefore working on current directory
        parent_dir = ".";
        filename = training_file;
    }
    size_t dot_index = filename.find_last_of("."); 
    string basename = filename.substr(0, dot_index);

    // Put global_bias into array in order to use generalized writeToFile
    float *global_bias_array = new float[1];
    global_bias_array[0] = global_bias;

    // Write components to file
    writeToFile(parent_dir, basename, "csv", "p", P, rows, cfg->n_factors, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "q", Q, cols, cfg->n_factors, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "user_bias", user_bias, rows, 1, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "item_bias", item_bias, cols, 1, cfg->n_factors);
    writeToFile(parent_dir, basename, "csv", "global_bias", global_bias_array, 1, 1, cfg->n_factors);

    // Free memory
    delete cfg;
    delete train_matrix;
    delete test_matrix;
    delete [] P;
    delete [] Q;
    delete [] losses;
    delete [] user_bias;
    delete [] item_bias;
    delete [] global_bias_array;
}



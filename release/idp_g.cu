// Copyright (C) 2013 IDP-G Team
// This file is part of IDP-G.
// 
// IDP-G is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// IDP-G is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with IDP-G.  If not, see <http://www.gnu.org/licenses/>.

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

#include <string>
#include <vector>
using namespace std;

#include "common.cuh"
#include "answer_db.cuh"

#define RECOVER_CSG 0

__host__ __device__ int choose(int n, int k) {
    if (n < k)
        return 0;
    k = min(k, n-k);
    int result = 1;
    for (int i = 1; i <= k; i++)
        result = result*(n-i+1)/i;
    return result;
}

__host__ __device__ unsigned int nth_mask(int n, int k, int id) {
    unsigned int result = 0;
    while (n > 0) {
        result = result << 1;
        if (k > 0) {
            int c = choose(n-1, k);
            if (id >= c) {
                result |= 1;
                k--;
                id -= c;
            }
        }
        n--;
    }
    return result;
}

#define gpu_bc(x) __popc(x)
#define cpu_bc(x) __builtin_popcount(x)

// Device code
__global__ void kernel(float *result, int n, int k)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= choose(n, k))
        return ;
    int m = nth_mask(n, k, id);
    float best = result[m];
    for (unsigned int m2 = m-1 & m; m2 > 0; m2 = m2-1 & m) {
        int m2_bc = gpu_bc(m2);
        if (k-m2_bc > m2_bc)
            continue;
        if (m2_bc > n - k && m != (1<<n)-1)
            continue;
        // if version of max slows down very slightly (4.70 vs 4.65)
        best = max(best, result[m2] + result[m^m2]);
    }
    result[m] = best;
}

void test_host(float *result, int n, u64 *p_start, u64 *p_end) {
    *p_start = now();
    for (unsigned int m = 0; m < (1<<n); m++) {
        int k = __builtin_popcount(m);
        if (2*n < 3*k && k < n)
            continue;

        float best = result[m];
        for (unsigned int m2 = m-1 & m; m2 > 0; m2 = m2-1 & m) {
            int m2_bc = cpu_bc(m2);
            if (k-m2_bc > m2_bc)
                continue;
            if (m2_bc > n - k && m != (1<<n)-1)
                continue;
            float nv = result[m2] + result[m^m2];
            if (nv > best)
                best = nv;
        }
        result[m] = best;
    }
    *p_end = now();
}

int gThreadsPerBlock;

void test_cuda(float *result, int n, u64 *p_start, u64 *p_end) {
    float *d_result;
    checkCudaErrors(cudaMalloc(&d_result, (1<<n)*sizeof(*d_result)));

    *p_start = now();
    // Copy vectors from host memory to device memory
    checkCudaErrors(cudaMemcpy(d_result,
                               result,
                               (1<<n)*sizeof(*d_result),
                               cudaMemcpyHostToDevice));
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    // Invoke kernel
    int threadsPerBlock = gThreadsPerBlock;
    
    for (int k = 2; k <= n; k++) {
        if (k > (n+1) / 2  && k != n)
            continue;
        int c = choose(n, k);
        int blocksPerGrid =
            (c + threadsPerBlock - 1) / threadsPerBlock;
        printf("blocks: %d, threads: %d\n", blocksPerGrid, threadsPerBlock);
        kernel<<<blocksPerGrid, threadsPerBlock, 0, stream[0]>>>(d_result, n, k);
        int k2 = n - k + 1;
        if (!(2*n < 3*k2  && k2 < n) && k != n) {
            kernel<<<blocksPerGrid, threadsPerBlock, 0, stream[1]>>>(d_result, n, k2);
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    checkCudaErrors(cudaMemcpy(result, d_result, (1<<n)*sizeof(*result), cudaMemcpyDeviceToHost));
    *p_end = now();

    // Free device memory
    for (int i = 0; i < 2; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_result);
}

void do_test(const string &method, void(*f)(float*,int,u64*,u64*), float *result,
        int n, int seed, float *v_for_verification_only) {
    u64 start, end;
    f(result, n, &start, &end);
    u64 recover_start = now();
    vector<int> solution = csg_recover(result, n);
    u64 recover_end = now();
    if (v_for_verification_only &&
        !csg_verify(v_for_verification_only, n, solution, result[(1<<n)-1])) {
        fprintf(stderr, "Test %4s: solution failed verification\n",
            method.c_str());
        assert(false);
    }
    bool verify_ok =
        AnswerDb().CheckAnswer(method, n, seed, result[(1<<n) - 1]);
    printf("Test %4s n=%d tpb=%d: %10lld us. Recover: %10lld us (%e). Verify: %s\n",
            method.c_str(),
            n,
            gThreadsPerBlock,
            end-start,
            recover_end-recover_start,
            1.0*(recover_end-recover_start) / (end-start),
            verify_ok ? "passed" : "failed");
}

void *memdup(void *src, int sz) {
    void *result = (void *)malloc(sz);
    memcpy(result, src, sz);
    return result;
}

void usage(const char *argv[]) {
    printf("Usage: %s <gpu|cpu> <n> [threads_per_block] [seed]\n", argv[0]);
    printf("                                                              \n");
    printf("gpu - IDP_G implementation                                    \n");
    printf("cpu - efficient IDP implementation.                           \n");
    printf("n - number of agents                                          \n");
    printf("threads_per_block - CUDA variable, max value: 1024            \n");
    printf("seed - used to generate random input for IDP_G, default: 1234 \n");
    printf("       In case you use seed different than default, the result\n");
    printf("       won't be verified, unless you add it to answers.ssv.   \n");
}

// Host code
int main(int argc, const char *argv[]) {
    if (argc < 3) {
        usage(argv);
        return (1);
    }
    const string &method = argv[1];
    int n = atoi(argv[2]);
    gThreadsPerBlock = 3 < argc ? atoi(argv[3]) : 1024;
    int seed = 4 < argc ? atoi(argv[4]) : 1234;

    MyRandom my_random;
    my_random.SetSeed(seed);
    int sz = (1<<n)*sizeof(float);
    // Allocate input vectors h_A and h_B in host memory
    float *v = (float*)malloc(sz);
    for (int i = 0; i < (1<<n); i++) {
        v[i] = i ? my_random.GetFloat() : 0;
    }

#if RECOVER_CSG
    float *result = (float*)memdup(v, sz);
#else
    float *result = v;
    v = NULL;
#endif

    if (method == "gpu")
        do_test("gpu", test_cuda,  result,  n, seed, v);
    else if (method == "cpu")
        do_test("cpu", test_host, result, n, seed, v);
    else {
        printf("Unknown method '%s'\n", method.c_str());
        usage(argv);
        assert(false);
    }

    return 0;
}

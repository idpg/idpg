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

#include "common.cuh"
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

using std::vector;

void __checkCudaErrors(cudaError err, const char *file, const int line) {
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ));
        throw 1;
    }
}

inline void __getLastCudaError(const char *errorMessage,
                               const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        printf("%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", 
            file, line, errorMessage, (int)err, cudaGetErrorString(err));
    }
}

vector<int> csg_recover(float *v, int n) {
    vector<int> result;
    vector<int> stack;
    stack.push_back((1<<n)-1);
    while (!stack.empty()) {
        int m = stack.back();
        stack.pop_back();
        float m_val = v[m];
        int best_m2 = -1;
        for (unsigned int m2 = m-1 & m; m2 > 0; m2 = m2-1 & m) {
            if (v[m2] + v[m^m2] == m_val)
                best_m2 = m2;
        }
        if (best_m2 == -1) {
            result.push_back(m);
            continue;
        }
        stack.push_back(m^best_m2);
        stack.push_back(best_m2);
    }
    return result;
}

bool csg_verify(float *v, int n, const vector<int> &solution, float answer) {
    int bit_sum = 0;
    float v_sum = 0;
    for (int i = 0; i < solution.size(); i++) {
        int m = solution[i];
        if (!m) {
            fprintf(stderr, "Solution contains en empty set!\n");
            return false;
        }
        if (bit_sum&m) {
            fprintf(stderr, "Solution parts overlap!\n");
            return false;
        }
        bit_sum |= m;
        v_sum += v[m];
    }
    if (bit_sum != (1<<n)-1) {
        fprintf(stderr, "Solution does not sum to the full set\n");
        return false;
    }
    if (fabsf(v_sum-answer) > 1e-4) {
        fprintf(stderr, "Solution gives answer %f where expected: %f\n",
            v_sum, answer);
        return false;
    }
    return true;
}


u64 now() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000000LL + tv.tv_usec;
}

void init_seed() {
    srand((now()>>32 & 0xffffffff) | (now() & 0xffffffff));
}

MyRandom::MyRandom() {
    timeval tv;
    gettimeofday(&tv, NULL);
    SetSeed((now() >> 32) ^ (now() ^ 0xffffffff));
}
void MyRandom::SetSeed(int seed) {
    w_ = seed;
    z_ = seed;
}
unsigned int MyRandom::GetU32() {
    z_ = 36969 * (z_ & 65535) + (z_ >> 16);
    w_ = 18000 * (w_ & 65535) + (w_ >> 16);
    unsigned int r = (z_ << 16) + w_;
    return r;
}
unsigned int MyRandom::GetU32(unsigned int limit) {
    unsigned int v = GetU32();
    return (unsigned long long)v * limit / 0x100000000ULL;
}
float MyRandom::GetFloat() {
    float res = (float)GetU32() / 0x100000000LL;
    assert (res >= 0 || res < 1.0);
    return res;
}

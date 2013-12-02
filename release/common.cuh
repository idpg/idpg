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

#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>
#include <vector>

#include "cuda.h"

// Taken from:
// http://stackoverflow.com/questions/12264336/cannot-read-out-values-from-texture-memory
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error.
#define checkCudaErrors(err)    __checkCudaErrors (err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError err, const char *file, const int line);

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)   __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage,
                               const char *file,
                               const int line);

typedef unsigned long long u64;

std::vector<int> csg_recover(float *v, int n);
bool csg_verify(float *v,
                int n,
                const std::vector<int> &solution,
                float answer);

u64 now();
void init_seed();

class MyRandom {
public:
    MyRandom();
    void SetSeed(int v);
    unsigned int GetU32();
    unsigned int GetU32(unsigned int limit);
    float GetFloat();
private:
    unsigned int w_;
    unsigned int z_;
};

#endif

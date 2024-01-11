// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381-fp2.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377-fp2.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#else
# error "no FEATURE"
#endif

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <msm/pippenger.cuh>

#include <cuda_runtime.h>
#include <iostream>
#include "util/logging.hpp"

 
#include <iostream>
#include <fstream>
using namespace std;


#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << endl;
        cerr << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void writeData(const affine_t points[], const scalar_t scalars[], size_t npoints) {
    // NOTE: one affine point takes 106 bytes, one scalar takes 30 bytes
    ofstream file_obj;
    string fname = "./input" + to_string(npoints) + ".dat";
    file_obj.open(fname, ios::out | ios::binary);
    for (unsigned i = 0; i < npoints; i++) {
        file_obj.write((char*)&scalars[i], sizeof(scalars[i]));
        file_obj.write((char*)&points[i], sizeof(points[i]));
    }
    file_obj.close();
}


tuple<scalar_t*, affine_t*> readData(size_t npoints) {
    ifstream file_obj;
    scalar_t* scalars = new scalar_t[npoints];
    affine_t* points = new affine_t[npoints];
    string fname = "./input" + to_string(npoints) + ".dat";
    file_obj.open(fname, ios::out | ios::binary);
    for (unsigned i = 0; i < npoints; i++) {
        file_obj.read((char*)&scalars[i], sizeof(scalars[i]));
        file_obj.read((char*)&points[i], sizeof(points[i]));
    }
    file_obj.close();
    return make_tuple(scalars, points);
}


extern "C"
RustError::by_value mult_pippenger_inf(point_t* out, const affine_t points[],
                                       size_t npoints, const scalar_t scalars[],
                                       size_t ffi_affine_sz)
{
    DEBUG_PRINTF("mult_pippenger_inf: calling <mult_pippenger>...\n");
    //writeData(points,scalars,npoints);
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);
}


#if defined(FEATURE_BLS12_381) || defined(FEATURE_BLS12_377)
typedef jacobian_t<fp2_t> point_fp2_t;
typedef xyzz_t<fp2_t> bucket_fp2_t;
typedef bucket_fp2_t::affine_inf_t affine_fp2_t;

extern "C"
RustError::by_value mult_pippenger_fp2_inf(point_fp2_t* out, const affine_fp2_t points[],
                                           size_t npoints, const scalar_t scalars[],
                                           size_t ffi_affine_sz)
{
    return mult_pippenger<bucket_fp2_t>(out, points, npoints, scalars, false, ffi_affine_sz);
}

__global__
void testInv(scalar_t* scalars, size_t npoints) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < npoints) {
        scalar_t out = scalars[tid];
        out = out.reciprocal();
    };
}

__global__
void testPoints(affine_t* points, size_t npoints) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < npoints) {
        //fp_t::one() / points[tid].get_X();
        //affine_t out = points[tid];
        //out = out.reciprocal();
    }; 
}

__global__
void testPointAdd(affine_t* points, size_t npoints) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < npoints) {
        affine_t p = points[tid];
        bucket_t b;
        b.inf();
        //b.add(p);
        //fp_t::one() / points[tid].get_X();
        //affine_t out = points[tid]+points[tid];
        //out = out.reciprocal();
    }; 
}

extern "C"
void mult_pippenger_toy(//point_t* out,
                                       size_t npoints
                                       //size_t ffi_affine_sz
                                    )
{
    //writeData(points, scalars, npoints);
    DEBUG_PRINTF("pippenger_inf.cu:mult_pippenger_toy npoints=%d\n", npoints);
    scalar_t* scalars = new scalar_t[npoints];
    affine_t* points = new affine_t[npoints];
    tie(scalars, points) = readData(npoints);
    DEBUG_PRINTF("pippenger_inf.cu:mult_pippenger_toy data read.\n");

    scalar_t* d_scalars;
    size_t nBytes = npoints * sizeof(scalar_t);
    DEBUG_PRINTF("pippenger_inf.cu:mult_pippenger_toy copying=%d bytes.\n", nBytes);
    cudaMalloc((scalar_t**)&d_scalars, nBytes);
    cudaMemcpy(d_scalars,(scalar_t *)points, nBytes, cudaMemcpyHostToDevice);

    affine_t* d_points;
    nBytes = npoints * sizeof(affine_t);
    DEBUG_PRINTF("pippenger_inf.cu:mult_pippenger_toy copying=%d bytes.\n", nBytes);
    cudaMalloc((affine_t**)&d_points, nBytes);
    cudaMemcpy(d_points,(affine_t *)points, nBytes, cudaMemcpyHostToDevice);

    CHECK_LAST_CUDA_ERROR();

    testPoints<<<ceil(npoints/256), 256>>>(d_points, npoints);
    //testPointAdd<<<ceil(npoints/256), 256>>>(d_points, npoints);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    affine_t p = points[npoints-1];
    bucket_t b;
    //b.add(p);


    //testInv<<<ceil(npoints/256), 256>>>(d_scalars, npoints);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    // testing
    scalar_t s1 = scalars[0] - scalars[0];
    scalar_t s2 = scalars[2].reciprocal() * scalars[2];
    if (s1.is_zero() != 1) {
        cerr << "Error";
    };
    if (s2.is_one() != 1) {
        // cerr << "Error"; FIXME: something very wrong
    };
    //return mult_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);
}


extern "C"
void transfer_points_cuda(const affine_t points[], size_t npoints)
{
    size_t nBytes = npoints * sizeof(affine_t);
    affine_t *d_points;
    cudaMalloc((affine_t**)&d_points, nBytes);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    DEBUG_PRINTF("transfer_points_cuda: allocating %d\n",nBytes);
    cudaMemcpy(d_points,(affine_t *)points,nBytes,cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();
    cudaFree(d_points);
}

#endif


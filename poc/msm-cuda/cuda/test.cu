#include <cuda.h>
#include <assert.h>


#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381-fp2.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377-fp2.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#else
# error "no FEATURE"
#endif

typedef fr_t scalar_t;

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#define SPPARK_DONT_INSTANTIATE_TEMPLATES

#include <iostream>
#include <fstream>
#include <tuple>
using namespace std;

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

__global__ void testInv(scalar_t* scalars, scalar_t* out, size_t npoints) 
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < npoints) {
        out[tid] = scalars[tid].reciprocal();
    };
}

__global__ void testOnCurve(const affine_t* points, fp_t* outs, size_t npoints) 
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < npoints) {
        affine_t p = points[tid];
        fp_t x = p.get_X();
        //fp_t y = p.get_Y();
        //const fp_t const1 = fp_t::mem_t::one();
        //fp_t r = const1*const1 - const1*const1;
        //assert( r.is_zero() );
        outs[tid] = x;
    };
}



__global__ void testCopyKernel(const fp_t* x, fp_t* out, size_t npoints) {
    size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < npoints) {
        fp_t const1 = fp_t::one();
        out[tid] = x[tid]+const1;
    }
}


int main() {
    // read data
    size_t npoints = 1<<23;
    const scalar_t* scalars = new scalar_t[npoints];
    const affine_t* data_points = new affine_t[npoints];
    tie(scalars, data_points) = readData(npoints);

    // transfer to device
    scalar_t* d_scalars;
    size_t nBytes = npoints * sizeof(scalar_t);
    cudaMalloc((void**)&d_scalars, nBytes);
    cudaMemcpy(d_scalars, scalars, nBytes, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    npoints = 1<<10;
    affine_t* d_points;
    nBytes = npoints * sizeof(affine_t);
    affine_t* points = new affine_t[npoints];
    for (int i=0; i<npoints; i++) {
        points[i] = data_points[i];
        affine_t p = points[i];
        fp_t x = p.get_X();
        fp_t y = p.get_Y();
        fp_t const1 = fp_t::one();
        fp_t r = (y*y)-(x*x*x)-(const1+const1+const1+const1);
        if( r.is_zero() != 1 ){
            printf("Bad %d",i);
        }
        //assert( r.is_zero() );
    }
    cudaMalloc((void**)&d_points, nBytes);
    cudaMemcpy(d_points, points, nBytes, cudaMemcpyHostToDevice);
    CHECK_LAST_CUDA_ERROR();

    printf("Testing curve on GPU...");
    npoints = 1<<10;
    fp_t* d_outs;
    fp_t outs[npoints];
    nBytes = sizeof(fp_t) * npoints;
    cudaMalloc((void**)&d_outs, nBytes);
    testOnCurve<<<npoints/256, 256>>>(d_points, d_outs, npoints);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(&outs, d_outs, nBytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    for (int i = 0; i<npoints; i++) {
        if( (outs[i]-points[i].get_X()).is_zero() != 1 ){
            printf("Bad %d",i);
        }
    }

    // run test kernel

    const int N = 10000;
    fp_t* x = new fp_t[N];
    fp_t* out = new fp_t[N];
    nBytes = N*sizeof(fp_t);
    fp_t* x_d;
    cudaMalloc((void**)&x_d, nBytes);
    fp_t* out_d;
    cudaMalloc((void**)&out_d, nBytes);

    cudaMemcpy(x_d, x, nBytes, cudaMemcpyHostToDevice);
    testCopyKernel<<<ceil(N/256), 256>>>(x_d, out_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(out, out_d, nBytes, cudaMemcpyDeviceToHost);
    CHECK_LAST_CUDA_ERROR();

    for (int i=0; i<100; i++) {
        fp_t res = out[i] - x[i] - fp_t::one();
        assert( res.is_zero() );
    }


    // test data
    printf("Testing curve on host...");
    npoints = 1<<10;
    for (size_t i = 0; i<npoints; i++) {
        affine_t p = points[i];
        fp_t x = p.get_X();
        fp_t y = p.get_Y();
        fp_t const1 = fp_t::one();
        fp_t r = (y*y)-(x*x*x)-(const1+const1+const1+const1);
        assert( r.is_zero() );
    }

    // test data
    printf("Testing curve on host...");
    npoints = 1<<10;
    for (size_t i = 0; i<npoints; i++) {
        affine_t p = points[i];
        fp_t x = p.get_X();
        fp_t y = p.get_Y();
        fp_t const1 = fp_t::one();
        fp_t r = (y*y)-(x*x*x)-(const1+const1+const1+const1);
        assert( r.is_zero() );
    }

}

// CUDA_ARCH="-arch=sm_80 -gencode arch=compute_70,code=sm_70 -t0"
// nvcc $CUDA_ARCH -std=c++17 -DFEATURE_BLS12_381 -D__ADX__ -I../.. -I../../deps/blst/src -L../../deps/blst -lblst cuda/test.cu -o test 

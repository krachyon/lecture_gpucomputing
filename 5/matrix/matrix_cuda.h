#pragma once
#include <cstdint>
#include "matrix.h"

// This is a really messy way of doing this: We can only include this file in pure C++ if there's no cuda syntax.
// but also, the caller can't instantiate a template then. So therefore the overloads....
// The other way is to declare a template here, implement and instantiate that explicitly in the source file,
// but then you only get the errors at link time if the instantiation is not there...


Matrix<float> mmul_cuda_naive (Matrix<float> const& left, Matrix<float> const& right);
Matrix<double> mmul_cuda_naive (Matrix<double> const& left, Matrix<double> const& right);
Matrix<int16_t> mmul_cuda_naive (Matrix<int16_t> const& left, Matrix<int16_t> const& right);

Matrix<float> mmul_cuda_shared(Matrix<float> const& left, Matrix<float> const& right);
Matrix<double> mmul_cuda_shared (Matrix<double> const& left, Matrix<double> const& right);

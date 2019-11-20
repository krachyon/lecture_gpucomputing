#pragma once
#include "matrix.h"

// This is a really messy way of doing this: We can only include this file in pure C++ if there's no cuda syntax.
// but also, the caller can't instantiate a templates then. So therefore the overloads.
// The other way is to declare a template and instantiate that explicitly in the source file, but then you only get
// the errors at link time...


Matrix<float> mmul_cuda_naive_float (Matrix<float> const& left, Matrix<float> const& right);
Matrix<double> mmul_cuda_naive_double (Matrix<double> const& left, Matrix<double> const& right);


Matrix<float> mmul_cuda_shared_float (Matrix<float> const& left, Matrix<float> const& right);
Matrix<double> mmul_cuda_shared_double (Matrix<double> const& left, Matrix<double> const& right);

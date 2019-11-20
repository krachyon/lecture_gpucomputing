#pragma once
#include <cstddef>
#include <memory>
#include <cassert>
#include <iostream>

template<typename T>
class Matrix
{
public:
    // Constructors etc.
    //empty, uninitialized memory just get sizes right
    Matrix(size_t m, size_t n)
     : M{m}, N{n}
    ,_mem{new T[m*n]}
    {}

    //Allow Matrix<> a{{1,0},{0,1}}; syntax
    Matrix(std::initializer_list<std::initializer_list<T>> const& init)
    : M{init.size()}, N{init.begin()->size()}, _mem{new T[M*N]}
    {
        size_t i = 0;
        for(auto const& row: init) {
            assert(row.size()==N);
            for (auto const& element: row)
            {
                _mem[i] = element;
                ++i;
            }
        }
    }
    // No copies, only moves allowed. Also Matrix without size makes no sense
    Matrix() = delete;
    Matrix(Matrix<T> const& ) = delete;
    Matrix& operator=(Matrix<T>const&) = delete;
    Matrix(Matrix<T> &&) noexcept = default;
    Matrix& operator=(Matrix<T>&&) noexcept = default;

    //unique_ptr takes care of memory cleanup
    ~Matrix()
    {}

    //index operators, use braces instead of [] two allow two indices
    T& operator()(size_t row, size_t col)
    {
        assert(row<M);
        assert(col<N);
        return _mem[N*row+col];
    }
    T operator()(size_t row, size_t col) const
    {
        assert(row<M);
        assert(col<N);
        return T(_mem[N*row+col]);
    }

    // what you'd expect from an stl-container
    T* begin(){return &_mem[0];}
    T* end(){return &_mem[M*N];}

    // make aa zero-initialized matrix
    static Matrix<T> zeros(size_t m, size_t n){
        Matrix<T> ret(m,n);
        std::fill(ret.begin(),ret.end(),0);
        return ret;
    }

    //not naminng it size() to avoid confusion
    size_t memsize() const
    {
        return M*N*sizeof(_mem);
    }
    size_t size() const
    {
        return M*N;
    }

    // similar to std::vector::data(). Be aware that you have to ensure that the matrix still exists when using this.
    // so treat it like a bomb
    T* data()
    {
        return _mem.get();
    }

    T* data() const
    {
        return _mem.get();
    }

    //members. These should be getters/setters by the book, but a civilised language should have properties...
    size_t M;
    size_t N;

    //storage
private:
    std::unique_ptr<T[]> _mem;
};

template<typename T>
Matrix<T> mmul(Matrix<T> const& left, Matrix<T> const& right)
{
    //check if dimensions are compatible
    size_t product_size = left.N; //size of scalar product == l.N == r.M
    assert(left.N == right.M);
    //output matrix has as many rows as left and as many columns as right
    size_t m = left.M;
    size_t n = right.N;

    Matrix<T> ret(m, n);

    for (size_t row=0; row<m; ++row)
        for(size_t col=0; col<n; ++col) {
            // do scalar product
            T elem = 0;
            for (size_t i=0; i<product_size; ++i) {
                elem += left(row, i)*right(i, col);
                //std::cout << "c(" << row << "," << col << ")+="
                //          << "a(" << row << ", " << i << ") * b("
                //          << i << ", " << col << ")" << std::endl;
            }
            ret(row,col) = elem;
        }
    return ret;
}





template<typename T>
std::ostream& operator<< (std::ostream& stream, const Matrix<T>& matrix)
{
    for (size_t row = 0; row!=matrix.M; ++row) {
        for (size_t col = 0; col!=matrix.N; ++col)
            stream << matrix(row,col) << " ";
        stream << std::endl;
    }
    return stream;
}


//create a matrix whose entries are M[i,j] = i+j
Matrix<float> make_ij_sum(size_t N)
{
    Matrix<float> ret(N, N);
    for (size_t row = 0; row!=ret.M; ++row)
        for (size_t col = 0; col!=ret.N; ++col) {
            ret(row,col) = row+col;
        }
    return ret;
}

//create a matrix whose entries are M[i,j] = i*j
Matrix<float> make_ij_product(size_t N)
{
    Matrix<float> ret(N, N);
    for (size_t row = 0; row!=ret.M; ++row)
        for (size_t col = 0; col!=ret.N; ++col) {
            ret(row,col) = row*col;
        }
    return ret;
}
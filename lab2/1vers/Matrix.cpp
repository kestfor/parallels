#include "Matrix.h"
#include <stdexcept>

Vector Matrix::operator*(Vector &v) {
    if (v.dimension != this->amountColumns) {
        throw std::runtime_error("different dimensions");
    }
    Vector res = Vector(v.dimension);
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < amountRows; i++) {
        long double sum = 0;
        for (int j = 0; j < amountColumns; j++) {
            sum += array[i][j] * v[j];
        }
        res[i] = sum;
    }
    return res;
}

Vector &Matrix::operator[](int index) const {
    if (array == nullptr) {
        throw std::runtime_error("access to null matrix");
    }
    if (index < 0 || index > amountRows) {
        throw std::range_error("invalid index");
    }
    return array[index];
}

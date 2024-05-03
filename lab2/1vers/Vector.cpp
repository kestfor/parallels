#include "Vector.h"
#include <stdexcept>
#include <cmath>
#include <omp.h>
#include <iostream>

Vector &Vector::operator+=(const Vector &v) {
    if (v.dimension != this->dimension) {
        throw std::runtime_error("different dimensions");
    }
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < this->dimension; i++) {
        this->array[i] += v.array[i];
    }
    return *this;
}

Vector &Vector::operator-=(const Vector &v) {
    if (v.dimension != this->dimension) {
        throw std::runtime_error("different dimensions");
    }
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < this->dimension; i++) {
        this->array[i] -= v.array[i];
    }
    return *this;
}

long double Vector::length() {
    long double res = 0;
    #pragma omp parallel for schedule(runtime) reduction(+:res)
    for (int i = 0; i < this->dimension; i++) {
        res += this->array[i] * this->array[i];
    }
    return std::sqrt(res);
}

void Vector::resize(int newDimension) {
    delete[] array;
    array = new long double[newDimension];
    this->dimension = newDimension;
}

long double &Vector::operator[](int index) {
    if (index < 0 || index >= dimension) {
        throw std::range_error("invalid index");
    }
    if (array == nullptr) {
        throw std::runtime_error("index access to null vector");
    }
    return this->array[index];
}

Vector &Vector::operator*=(const long double val) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < this->dimension; i++) {
        this->array[i] *= val;
    }
    return *this;
}



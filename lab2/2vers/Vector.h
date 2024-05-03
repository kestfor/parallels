#ifndef BPP_VECTOR_H
#define BPP_VECTOR_H
#include <cstring>
#include <utility>
using std::pair;

class Vector {
protected:
    long double *array;
    int dimension;

public:
    friend class Matrix;

    Vector() : dimension(0) {
        array = nullptr;
    }

    Vector(Vector &v) : Vector(v.dimension) {
        memcpy(this->array, v.array, v.dimension * sizeof(long double));
    }

    explicit Vector(int dimension, long double setValue=0) : dimension(dimension) {
        array = new long double[dimension];
        for (int i = 0; i < dimension; i++) {
            array[i] = setValue;
        }
    }

    Vector& operator=(const Vector &v) {
        if (&v ==  this) {
            return *this;
        }
        this->dimension = v.dimension;
        delete[] array;
        this->array = new long double[dimension];
        memcpy(this->array, v.array, sizeof(long double) * dimension);
        return *this;
    }

    ~Vector() {
        delete[] array;
    }

    long double& operator[](int index) const;

    Vector& operator +=(const Vector &v);

    Vector& operator -=(const Vector &v);

    Vector& operator *=(long double val);

    void resize(int newDimension);

    long double length();
};

#endif //BPP_VECTOR_H

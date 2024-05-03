#include <iostream>
#include "Matrix.h"
#include "Vector.h"
#include <cmath>
#include <omp.h>
#include <chrono>

using std::chrono::high_resolution_clock;

struct Context {
    Matrix A;
    Vector x;
    Vector b;
    Vector v;
    long double t;
    const long double error;
    const int size;

    Context(const int size, const double error) : size(size), error(error) {
        A = Matrix(size, size, 1);
        v = Vector(size);
        x = Vector(size, 0);
        b = Vector(size, size + 1);
        t = 0.0001;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    A[i][j] = 2.0;
                }
            }
        }
    }
};

bool isCloseEnough(Context &cont) {
    cont.A.partialMult(cont.x, cont.v);
    cont.v -= cont.b;
    long double res = cont.v.length() / cont.b.length();
    return res < cont.error;
}

void next(Context &cont) {
    cont.A.partialMult(cont.x, cont.v);
    cont.v -= cont.b;
    cont.v *= cont.t;
    cont.x -= cont.v;
}

int main(int argc, char *argv[]) {
    int n = 10000;
    const double error = 10e-19;
    Context cont = Context(n, error);
    high_resolution_clock::time_point begin = high_resolution_clock::now();
    #pragma omp parallel
    {
        while (!isCloseEnough(cont)) {
            next(cont);
        }
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

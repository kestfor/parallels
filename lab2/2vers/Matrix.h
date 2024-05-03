#ifndef BPP_MATRIX_H
#define BPP_MATRIX_H
#include "Vector.h"


class Matrix {
private:
    int amountRows;
    int amountColumns;
    Vector *array;

public:

    Matrix() {
        amountRows = 0;
        amountColumns = 0;
        array = nullptr;
    }

    Matrix(int rows, int columns, int setValue=0) : amountColumns(columns), amountRows(rows) {
        array = new Vector[rows];
        for (int i = 0; i < rows; i++) {
            array[i] = Vector(columns, setValue);
        }
    }

    Matrix(const Matrix &matrix) : amountRows(matrix.amountRows), amountColumns(matrix.amountColumns) {
        array = new Vector(amountRows);
        for (int i = 0; i < amountRows; i++) {
            array[i] = matrix[i];
        }
    }

    Matrix& operator=(const Matrix &matrix) {
        if (&matrix == this) {
            return *this;
        }
        delete[] array;
        this->amountRows = matrix.amountRows;
        this->amountColumns = matrix.amountColumns;
        array = new Vector[amountRows];
        for (int i = 0; i < amountRows; i++) {
            array[i] = matrix[i];
        }
        return *this;
    }

    Vector& operator[](int index) const;

    Vector operator *(Vector &v);

    int getAmountRows() const {
        return amountRows;
    }

    int getAmountColumns() const {
        return amountColumns;
    }

    void partialMult(Vector &v, Vector &res);
};


#endif //BPP_MATRIX_H

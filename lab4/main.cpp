#include <stdexcept>
#include <exception>
#include <cstring>
#include <mpi.h>
#include <vector>
#include <chrono>
using namespace std::chrono;

using std::max;

#define a 10000
#define Nx 128
#define Ny 128
#define Nz 128
#define Dx 2
#define Dy 2
#define Dz 2
#define error 10e-8
#define start_x (-1)
#define start_y (-1)
#define start_z (-1)

const double Hx = Dx / (Nx - 1.0);
const double Hy = Dy / (Ny - 1.0);
const double Hz = Dz / (Nz - 1.0);

int size = 1;
int rank = 0;


std::vector<int> offsets = {};
std::vector<int> lengths = {};

double orig_func_val(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double orig_func_val(int i, int j, int k) {
    double x, y, z;
    x = start_x + i * Hx;
    y = start_y + j * Hy;
    z = start_z + (k + offsets[rank]) * Hz;
    return orig_func_val(x, y, z);
}

class DiscreteFunction {
private:
    double *values=nullptr;
    int size1;
    int size2;
    int size3;

    [[nodiscard]] bool is_valid_ind(int i, int j, int k) const {
        return (i < size1 && j < size2 && k < size3);
    }

public:
    DiscreteFunction(int hx, int hy, int hz) {
        this->size1 = hx;
        this->size2 = hy;
        this->size3 = hz;
        this->values = (double *) calloc(size1 * size3 * size2, sizeof(double));
    }

    DiscreteFunction(const DiscreteFunction &f) : DiscreteFunction(f.size1, f.size2, f.size3) {
        memcpy(this->values, f.values, size1 * size2 * size3 * sizeof(double));
    }

    DiscreteFunction& operator=(const DiscreteFunction &f) {
        if (&f == this) {
            return *this;
        } else {
            memcpy(this->values, f.values, size1 * size2 * size3 * sizeof(double));
            return *this;
        }
    }

    ~DiscreteFunction() {
        delete[] this->values;
    }

    [[nodiscard]] int get_first_size() const {
        return size1;
    }

    [[nodiscard]] int get_second_size() const {
        return size2;
    }

    [[nodiscard]] int get_third_size() const {
        return size3;
    }

    double get_value(int i, int j, int k) {
        if (is_valid_ind(i, j, k)) {
            return values[k * size1 * size2 + j * size1 + i];
        } else {
            char buff[256];
            sprintf(buff, "range error with indexes: %d, %d, %d on %d rank", i, j, k, rank);
            throw std::range_error(buff);
        }
    }

    double *get_array() {
        return values;
    }

    void set_value(double val, int i, int j, int k) {
        if (is_valid_ind(i, j, k)) {
            values[k * size1 * size2 + j * size1 + i] = val;
        } else {
            char buff[256];
            sprintf(buff, "range error with indexes: %d, %d, %d on %d rank", i, j, k, rank);
            throw std::range_error(buff);
        }
    }
};

double right_part(int i, int j, int k) {
    return 6 - a * orig_func_val(i, j, k);
}

double max_delta(DiscreteFunction &first, DiscreteFunction &second) {
    double max = -1;
    int s1 = first.get_first_size();
    int s2 = first.get_second_size();
    int s3 = first.get_third_size();
    for (int k = 0; k < s3; k++) {
        for (int i = 0; i < s1; i++) {
            for (int j = 0; j < s2; j++) {
                max = std::max(std::abs(first.get_value(i, j, k) - second.get_value(i, j, k)), max);
            }
        }
    }
    return max;
}

void set_border_values(DiscreteFunction &func) {
    int s1 = func.get_first_size();
    int s2 = func.get_second_size();
    int s3 = func.get_third_size();

    if (rank == 0) {
        for (int j = 0; j < s2; j++) {
            for (int i = 0; i < s1; i++) {
                func.set_value(orig_func_val(i, j, 0), i, j, 0);
            }
        }
    }

    if (rank == size - 1) {
        for (int j = 0; j < s2; j++) {
            for (int i = 0; i < s1; i++) {
                func.set_value(orig_func_val(i, j, s3 - 1), i, j, s3 - 1);
            }
        }
    }

    for (int k = 0; k < s3; k++) {
        for (int i = 0; i < s1; i++) {
            func.set_value(orig_func_val(i, 0, k), i, 0, k);
            func.set_value(orig_func_val(i, s2 - 1, k), i, s2 - 1, k);
        }
    }

    for (int k = 0; k < s3; k++) {
        for (int j = 0; j < s2; j++) {
            func.set_value(orig_func_val(0, j, k), 0, j, k);
            func.set_value(orig_func_val(s1 - 1, j, k), s1 - 1, j, k);
        }
    }
}

bool is_close_enough(DiscreteFunction &first, DiscreteFunction &second) {
    double local_delta = max_delta(first, second);
    double global_delta;
    MPI_Allreduce(&local_delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_delta < error;
}

double get_value_using_formula(DiscreteFunction &func, int i, int j, int k) {
    return (1.0 / ((2 / (Hx * Hx)) + (2 / (Hy * Hy)) + (2 / (Hz * Hz))  + a)) *
           ((func.get_value(i + 1, j, k) + func.get_value(i - 1, j, k)) / (Hx * Hx) +
            (func.get_value(i, j + 1, k) + func.get_value(i, j - 1, k)) / (Hy * Hy) +
            (func.get_value(i, j, k + 1) + func.get_value(i, j, k - 1)) / (Hz * Hz) -
            right_part(i,  j, k));
}

void calculate_values_near_borders(DiscreteFunction &func) {
    for (int k = 1; k < func.get_third_size() - 1; k+=func.get_third_size() - 3) {
        for (int i = 1; i < func.get_first_size() - 1; i++) {
            for (int j = 1; j < func.get_second_size() - 1; j++) {
                func.set_value(get_value_using_formula(func, i, j, k), i, j, k);
            }
        }
    }
}

void next(DiscreteFunction &func) {

    calculate_values_near_borders(func);

    MPI_Request req_send_low, req_send_high, req_receive_low, req_receive_high;
    int data_num = func.get_second_size() * func.get_first_size();
    if (rank != 0) {
        MPI_Irecv(func.get_array(), data_num, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req_receive_low);
    }
    if (rank != size - 1) {
        MPI_Irecv(func.get_array() + data_num * (func.get_third_size() - 1), data_num, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req_receive_high);
    }
    if (rank < size - 1) {
        int rank_to = rank + 1;
        double *data_start = func.get_array() + data_num * (func.get_third_size() - 2);
        MPI_Isend(data_start, data_num, MPI_DOUBLE, rank_to, 0, MPI_COMM_WORLD, &req_send_high);
    }
    if (rank > 0) {
        int rank_to = rank - 1;
        double *data_start = func.get_array() + data_num;
        MPI_Isend(data_start, data_num, MPI_DOUBLE, rank_to, 0, MPI_COMM_WORLD, &req_send_low);
    }

    for (int k = 2; k < func.get_third_size() - 2; k++) {
        for (int i = 1; i < func.get_first_size() - 1; i++) {
            for (int j = 1;j < func.get_second_size() - 1; j++) {
                func.set_value(get_value_using_formula(func, i, j, k), i, j, k);
            }
        }
    }
    if (rank != 0) {
        MPI_Wait(&req_receive_low, MPI_STATUS_IGNORE);
    }
    if (rank != size - 1) {
        MPI_Wait(&req_receive_high, MPI_STATUS_IGNORE);
    }

}

void init_offsets(std::vector<int> &off, std::vector<int> &lens) {
    int remainder = Nz % size;
    int start = 0;
    for (int i = 0; i < size; i++) {
        int val = Nz / size + (i < remainder ? 1 : 0);
        if (i != 0) {
            start -= 2;
            val += 2;
        }
        off.push_back(start);
        lens.push_back(val);
        start += val;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    init_offsets(offsets, lengths);
    DiscreteFunction first(Nx, Ny, lengths[rank]);
    set_border_values(first);
    DiscreteFunction second(first);

    auto start = high_resolution_clock::now();

    next(first);
    while (!is_close_enough(first, second)) {
        second = first;
        next(first);
    }

    auto end = high_resolution_clock::now();

    DiscreteFunction res(Nx, Ny, lengths[rank]);
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < lengths[rank]; k++) {
                res.set_value(orig_func_val(i, j, k), i, j, k);
            }
        }
    }

    double local_delta = max_delta(first, res);
    double global_delta;
    MPI_Reduce(&local_delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("max_delta: %.*f\ntime: %lf\n", 10, global_delta, (double) duration_cast<milliseconds>(end - start).count() / 1000.0);
    }
    MPI_Finalize();
}
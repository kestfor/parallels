#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdbool.h>

#define NUM_DIMS 2

#define M 2000
#define N 2000
#define K 2000


void mult(int *matrix_sizes, double *A, double *B, double *C, int *computer_grid_sizes, MPI_Comm comm) {
    double *sub_matrix_A;
    double *sub_matrix_B;
    double *sub_matrix_C;
    int sub_matrix_sizes[2];
    int coords[2];
    int rank;
    int *receive_counts_for_gather_c = NULL;
    int *displacements_for_gather_c = NULL;
    int *send_count_b = NULL;
    int *displacements_for_scatter_b = NULL;

    MPI_Datatype type_c, types[2], type_b;
    int block_lengths[2];
    int periods[2], remains[2];

    MPI_Comm comm_2D, comm_1D[2];

    //раздаем данные о размерах
    MPI_Bcast(matrix_sizes, 3, MPI_INT, 0, comm);
    MPI_Bcast(computer_grid_sizes, 2, MPI_INT, 0, comm);

    periods[0] = 0;
    periods[1] = 0;

    //создаем декартову решетку
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, computer_grid_sizes, periods, false, &comm_2D);
    MPI_Comm_rank(comm_2D, &rank);
    MPI_Cart_coords(comm_2D, rank, NUM_DIMS, coords);

    //убираем i-ое измеренеие, получаем коммутаторы для broadcast
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            remains[j] = (i == j);
        }
        MPI_Cart_sub(comm_2D, remains, &comm_1D[i]);
    }

    sub_matrix_sizes[0] = matrix_sizes[0] / computer_grid_sizes[0];
    sub_matrix_sizes[1] = matrix_sizes[2] / computer_grid_sizes[1];

    sub_matrix_A = malloc(sub_matrix_sizes[0] * matrix_sizes[1] * sizeof(double));
    sub_matrix_B = malloc(matrix_sizes[1] * sub_matrix_sizes[1] * sizeof(double));
    sub_matrix_C = malloc(sub_matrix_sizes[0] * sub_matrix_sizes[1] * sizeof(double));

    if (rank == 0) {

        //создание вектора для scatter B
        MPI_Type_vector(matrix_sizes[1], sub_matrix_sizes[1], matrix_sizes[2], MPI_DOUBLE, &types[0]);
        long size;
        MPI_Type_extent(MPI_DOUBLE, &size);
        block_lengths[0] = 1;
        block_lengths[1] = 1;
        types[1] = MPI_UB;
        long *displacements = malloc(sizeof(long) * 2);
        displacements[0] = 0;
        displacements[1] = size * sub_matrix_sizes[1];
        MPI_Type_create_struct(2, block_lengths, displacements, types, &type_b);
        MPI_Type_commit(&type_b);

        displacements_for_scatter_b = malloc(computer_grid_sizes[1] * sizeof(int));
        send_count_b = malloc(computer_grid_sizes[1] * sizeof(int));
        for (int i = 0; i < computer_grid_sizes[1]; i++) {
            displacements_for_scatter_b[i] = i;
            send_count_b[i] = 1;
        }

        //создание вектора для gather C
        MPI_Type_vector(sub_matrix_sizes[0], sub_matrix_sizes[1], matrix_sizes[2], MPI_DOUBLE, &type_c);
        MPI_Type_create_struct(2, block_lengths, displacements, types, &type_c);
        MPI_Type_commit(&type_c);

        displacements_for_gather_c = malloc(computer_grid_sizes[0] * computer_grid_sizes[1] * sizeof(int));
        receive_counts_for_gather_c = malloc(computer_grid_sizes[0] * computer_grid_sizes[1] * sizeof(int));
        for (int i = 0; i < computer_grid_sizes[0]; i++) {
            for (int j = 0; j < computer_grid_sizes[1]; j++) {
                displacements_for_gather_c[i * computer_grid_sizes[1] + j] = (
                        i * computer_grid_sizes[1] * sub_matrix_sizes[0] + j);
                receive_counts_for_gather_c[i * computer_grid_sizes[1] + j] = 1;
            }
        }

        free(displacements);

    }

    if (coords[1] == 0) {
        MPI_Scatter(A, sub_matrix_sizes[0] * matrix_sizes[1], MPI_DOUBLE, sub_matrix_A,
                    sub_matrix_sizes[0] * matrix_sizes[1], MPI_DOUBLE, 0, comm_1D[0]);
    }


    if (coords[0] == 0) {
        MPI_Scatterv(B, send_count_b, displacements_for_scatter_b, type_b, sub_matrix_B,
                     matrix_sizes[1] * sub_matrix_sizes[1], MPI_DOUBLE, 0, comm_1D[1]);
    }


    MPI_Bcast(sub_matrix_A, sub_matrix_sizes[0] * matrix_sizes[1], MPI_DOUBLE, 0, comm_1D[1]);
    MPI_Bcast(sub_matrix_B, matrix_sizes[1] * sub_matrix_sizes[1], MPI_DOUBLE, 0, comm_1D[0]);

    int m = sub_matrix_sizes[1];
    int n = matrix_sizes[1];
    for (int i = 0; i < sub_matrix_sizes[0]; i++) {
        for (int j = 0; j < m; j++) {
            sub_matrix_C[i * m + j] = 0;
            for (int k = 0; k < n; k++) {
                sub_matrix_C[i * m + j] += sub_matrix_A[n * i + k] * sub_matrix_B[m * k + j];
            }
        }
    }

    MPI_Gatherv(sub_matrix_C, sub_matrix_sizes[0] * sub_matrix_sizes[1], MPI_DOUBLE, C, receive_counts_for_gather_c,
                displacements_for_gather_c, type_c, 0, comm_2D);

    free(sub_matrix_A);
    free(sub_matrix_B);
    free(sub_matrix_C);
    free(displacements_for_scatter_b);
    free(send_count_b);

    MPI_Comm_free(&comm_2D);


    for (int i = 0; i < 2; i++) {
        MPI_Comm_free(&comm_1D[i]);
    }

    if (rank == 0) {
        free(receive_counts_for_gather_c);
        free(displacements_for_gather_c);
        MPI_Type_free(&type_b);
        MPI_Type_free(&type_c);
        MPI_Type_free(&types[0]);
    }
}


int main(int argc, char *argv[]) {
    int size, rank, matrix_sizes[3], computer_grid_sizes[2];
    double *A = NULL, *B = NULL, *C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "empty grid size");
        }
        exit(1);
    }

    int p0 = atoi(argv[1]);
    int p1 = atoi(argv[2]);
    if (p0 == 0 || p1 == 0) {
        if (rank == 0) {
            fprintf(stderr, "invalid grid size");
        }
        exit(1);
    }

    if (size != p0 * p1) {
        if (rank == 0) {
            fprintf(stderr, "incompatible grid size");
        }
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) {
        matrix_sizes[0] = M;
        matrix_sizes[1] = N;
        matrix_sizes[2] = K;

        computer_grid_sizes[0] = p0;
        computer_grid_sizes[1] = p1;

        A = malloc(M * N * sizeof(double));
        B = malloc(N * K * sizeof(double));
        C = malloc(M * K * sizeof(double));

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                A[i * M + j] = 1;
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                B[i * N + j] = 1;
            }
        }

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                C[M * i + j] = 0;
            }
        }
    }

    mult(matrix_sizes, A, B, C, computer_grid_sizes, MPI_COMM_WORLD);

    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
}

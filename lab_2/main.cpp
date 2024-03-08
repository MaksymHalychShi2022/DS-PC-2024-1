#include <omp.h>
#include<iostream>
#include<vector>
#include<cmath>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

const int N = 10000;

void init(Matrix &A, Matrix &B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = cos(i + j);
            B[i][j] = sin(i + j);
        }
    }
}

void calculate(const Matrix &mat, Vector &res, omp_sched_t type, int chunk) {
    omp_set_schedule(type, chunk);
    #pragma omp parallel for schedule(runtime)
    for(int i = 0; i < N; i++) {
        int count = 0;
        for(int j = 0; j < N; j++) {
            if(mat[i][j] < 0) {
                count++;
            }
        }
        res[i] = count;
    }
}

int main() {
    auto A = Matrix(N, Vector(N)), B = Matrix(N, Vector(N));
    init(A, B);
    auto C = Vector(N), D = Vector(N);

    auto start = omp_get_wtime();
    calculate(B, D, omp_sched_static, 6);
    auto end = omp_get_wtime();
    auto time_static = end-start;
    std::cout << "Time for (static, 6) " << time_static << "\n";

    start = omp_get_wtime();
    calculate(B, D, omp_sched_dynamic, 4);
    end = omp_get_wtime();
    auto time_dynamic = end-start;
    std::cout << "Time for (dynamic, 4) " << time_dynamic << "\n\n";

    if (time_static > time_dynamic) {
        std::cout << "So we see, that (dynamic, 4) works better\n";
    } else {
        std::cout << "So we see, that (static, 6) works better\n";
    }
}
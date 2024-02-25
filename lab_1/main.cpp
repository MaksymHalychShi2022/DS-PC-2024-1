#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

struct ResultRecord {
    int n, ntreads;
    double seconds;
};

void write_csv(const std::vector<ResultRecord> &results, const char *file_name = "results.csv") {
    std::ofstream outFile(file_name);
    if (outFile.is_open()) {

        for (auto it: results) {
            outFile << it.n << ',' << it.ntreads << "," << it.seconds << "\n";
        }

        outFile.close();
    }
}

void init(double **A, double *b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = std::sqrt(i + 1);
        for (int j = 0; j < n; j++) {
            A[i][j] = i * std::log(j + 1);
        }
    }
}

void calculate_serial(double **a, double *b, double *res, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += a[i][j] * b[j];
        }
        res[i] = sum;
    }
}

void calculate_parallel(double **a, double *b, double *res, int n, int ntreads) {
#pragma omp parallel for num_threads(ntreads) schedule(static)
    for (int i = 0; i < n; i++) {
        double sum = 0;
//#pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; j++) {
            sum += a[i][j] * b[j];
        }
        res[i] = sum;
    }
}

const std::vector<int> NTREADS = {
        2, 4, 8, 16, 32
};

const std::vector<int> N = {
        80, 1000, 5000, 10000
};

int main() {
    auto results = std::vector<ResultRecord>();

    for (auto n: N) {
        auto A = new double *[n];
        for (int i = 0; i < n; ++i) {
            A[i] = new double[n];
        }
        auto b = new double[n], res = new double[n];
        init(A, b, n);

        auto start = omp_get_wtime();
        calculate_serial(A, b, res, n);
        auto end = omp_get_wtime();
        results.push_back({n, 1, end - start});

        for (auto ntreads: NTREADS) {
            start = omp_get_wtime();
            calculate_parallel(A, b, res, n, ntreads);
            end = omp_get_wtime();
            results.push_back({n, ntreads, end - start});
        }

        for (int i = 0; i < n; ++i) {
            delete[] A[i];
        }
        delete[] A;
        delete[] b;
        delete[] res;
    }


    write_csv(results, "../lab_1/results.csv");
    return 0;
}

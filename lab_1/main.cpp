#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

class Results {
private:
    struct ResultRecord {
        int n, ntreads;
        double seconds;
    };

    std::vector<ResultRecord> results;
public:
    Results() {
        results = std::vector<ResultRecord>();
    }

    void addRecord(int n, int ntreads, double seconds, bool logging=true){
        results.push_back({n, ntreads, seconds});
        if (logging) {
            std::cout << "n: " << n << "\tntreads: " << ntreads << "\tseconds: " << seconds << "\n";
        }
    }

    void writeToCSV(const char *fileName) {
        std::ofstream outFile(fileName);
        if (outFile.is_open()) {
            outFile << "n,ntreads,seconds\n";

            for (auto it: results) {
                outFile << it.n << ',' << it.ntreads << "," << it.seconds << "\n";
            }

            outFile.close();
            std::cout << "Results was written to lab_1/results.csv\n";
        } else {
            std::cout << "Error while writing to file " << fileName << "\n";
        }
    }
};

void init(Matrix &A, Vector &b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = std::sqrt(i + 1);
        for (int j = 0; j < n; j++) {
            A[i][j] = i * std::log(j + 1);
        }
    }
}

void calculate_serial(const Matrix &a, const Vector &b, Vector &res, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += a[i][j] * b[j];
        }
        res[i] = sum;
    }
}

void calculate_parallel(const Matrix &a, const Vector &b, Vector &res, int n, int ntreads) {
#pragma omp parallel for num_threads(ntreads) schedule(static) default(none) shared(a, b, res, n)
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += a[i][j] * b[j];
        }
        res[i] = sum;
    }
}

const std::vector<int> NTREADS = {
        2, 4, 8, 16, 32, 64, 128, 256, 512
};

const std::vector<int> N = {
        100, 2'000, 5'000, 10'000
};

int main() {
    auto results = Results();

    for (auto n: N) {
        auto A = Matrix(n, Vector(n));
        auto b = Vector(n), res = Vector(n);
        init(A, b, n);

        auto start = omp_get_wtime();
        calculate_serial(A, b, res, n);
        auto end = omp_get_wtime();
        results.addRecord(n, 1, end - start);

        for (auto ntreads: NTREADS) {
            start = omp_get_wtime();
            calculate_parallel(A, b, res, n, ntreads);
            end = omp_get_wtime();
            results.addRecord(n, ntreads, end - start);
        }
    }

    results.writeToCSV("../lab_1/results.csv");

    return 0;
}

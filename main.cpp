#include <atri/core/fwd.hpp>
#include <chrono>
#include <iostream>
extern void nice();
int main(int, char **) {
    // nice();
    atri::matrix m(3, 3);
    atri::matrix g;
    m.setRandom();
    // using namespace Eigen;
    // using namespace std;
    // // typedef bfloat16 scalar_t;
    // // typedef float scalar_t;
    // typedef double scalar_t;
    // std::vector<Matrix<scalar_t, 72, 72>> a;
    // for (int i = 0; i < 100; i ++){
    //     a.push_back(decltype(a)::value_type::Random());
    // }
    // Matrix<scalar_t, 72, 72> b;
    // Matrix<scalar_t, 72, 36> c;
    // c.setRandom();
    // LDLT<Matrix<scalar_t, 72, 72>> ldlt;
    // int N = 1000;
    // auto start = chrono::high_resolution_clock::now();
    // for (int i = N; i; i--) {
    //     for (int j = 0; j < 100; j++){
    //         b = a[j].transpose() * a[j];
    //         ldlt.compute(b);
    //         ldlt.solve(c);
    //     }
    // }
    // auto end = chrono::high_resolution_clock::now();
    // cout << chrono::duration_cast<chrono::microseconds>(end - start).count()
    // / N << " us";
    return 0;
}

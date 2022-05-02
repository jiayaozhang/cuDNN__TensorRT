#include <iostream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include "cpu.h"

using namespace std;

int main(int argc, char *argv[]) {

    int n = atoi(argv[1]);
    cout << n << endl;

    vector<float> vecA(n);
    vector<float> vecB(n);
    vector<float> vecC(n);


    struct timeval t1, t2;

    for (int i = 0; i < n; i++) {
        float a = rand() / double(RAND_MAX);
        float b = rand() / double(RAND_MAX);
        vecA[i] = a;
        vecB[i] = b;
    }

    gettimeofday(&t1, NULL);

    vecAdd(&vecA[0], &vecB[0], &vecC[0], vecA.size());

    gettimeofday(&t2, NULL);

    //for (int i = 0; i < 10; i++) 
    //    cout << vecA[i] << " " << vecB[i] << " " << vecC[i] << endl;
    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

    return 0;
}

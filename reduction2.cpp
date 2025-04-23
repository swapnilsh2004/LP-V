#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
#include <chrono>

using namespace std;
using namespace chrono;

vector<float> serialReduction(const vector<long long> &arr) {
    auto start_time = high_resolution_clock::now();
    int minVal = numeric_limits<int>::max();
    int maxVal = numeric_limits<int>::min();
    long long sum = 0;
    int n = arr.size();
    vector<float> time(4);
    
    for (int i = 0; i < n; i++) {
        if (arr[i] < minVal) minVal = arr[i];
    }
    auto end_time = high_resolution_clock::now();

    time[0] = (end_time - start_time).count();
    start_time = high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        if (arr[i] > maxVal) maxVal = arr[i];
    }
    end_time = high_resolution_clock::now();
    time[1] = (end_time - start_time).count();
    start_time = high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    end_time = high_resolution_clock::now();
    time[2] = (end_time - start_time).count();
    start_time = high_resolution_clock::now();
    double average = static_cast<double>(sum) / n;
    end_time = high_resolution_clock::now();
    time[3] = (end_time - start_time).count();
    
    cout << "Serial Minimum: " << minVal << endl;
    cout << "Serial Maximum: " << maxVal << endl;
    cout << "Serial Sum: " << sum << endl;
    cout << "Serial Average: " << average << endl;
    return time;
}

vector<float> parallelReduction(const vector<long long> &arr) {
    auto start_time = high_resolution_clock::now();
    int minVal = numeric_limits<int>::max();
    int maxVal = numeric_limits<int>::min();
    long long sum = 0;
    int n = arr.size();
    vector<float> time(4);
    
    #pragma omp parallel for reduction(min:minVal)
    for (int i = 0; i < n; i++) {
        if (arr[i] < minVal) minVal = arr[i];
    }
    auto end_time = high_resolution_clock::now();
    time[0] = (end_time - start_time).count();
    start_time = high_resolution_clock::now();

    #pragma omp parallel for reduction(max:maxVal) 
    for (int i = 0; i < n; i++) {
        if (arr[i] > maxVal) maxVal = arr[i];
    }
    end_time = high_resolution_clock::now();
    time[1] = (end_time - start_time).count();
    start_time = high_resolution_clock::now();

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    end_time = high_resolution_clock::now();
    time[2] = (end_time - start_time).count();
    start_time = high_resolution_clock::now();
    double average = static_cast<double>(sum) / n;
    end_time = high_resolution_clock::now();
    time[3] = (end_time - start_time).count();
    
    cout << "Parallel Minimum: " << minVal << endl;
    cout << "Parallel Maximum: " << maxVal << endl;
    cout << "Parallel Sum: " << sum << endl;
    cout << "Parallel Average: " << average << endl;
    return time;
}

int main() {
    vector<long long> arr(100000);
    for (int i = 0; i < arr.size(); i++) {
        arr[i] = rand() % 10000;
    }
    
    vector<float> v1 = serialReduction(arr);
    cout << "---------------------------" << endl;
    vector<float> v2 = parallelReduction(arr);
    cout<<v1[0]<<" "<<v1[1]<<" "<<v1[2]<<" "<<v1[3]<<"\n";
    cout<<v2[0]<<" "<<v2[1]<<" "<<v2[2]<<" "<<v2[3]<<"\n";
    cout<<"SPEED UP FACTOR "<<"\n";
    cout<<"min "<<v1[0]/v2[0]<<"\n";
    cout<<"max "<<v1[1]/v2[1]<<"\n";
    cout<<"sum "<<v1[2]/v2[2]<<"\n";
    
    return 0;
}

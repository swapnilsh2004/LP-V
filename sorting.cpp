#include <iostream>
#include <chrono>
#include <omp.h>
#include <vector>
using namespace std;
using namespace std::chrono;

// Sequential Bubble Sort
void bubble_sort(vector<int>& arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }

    for(int i=0;i<n;i++){
        cout<<arr[i]<<" ";
    }
}

// Sequential Merge Sort
void merge(vector<int>& arr, int start, int mid, int end) {
    vector<int> left(arr.begin() + start, arr.begin() + mid + 1);
    vector<int> right(arr.begin() + mid + 1, arr.begin() + end + 1);

    int i = 0, j = 0, k = start;
    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }
    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

void merge_sort(vector<int>& arr, int start, int end) {
    if (start < end) {
        int mid = start + (end - start) / 2;
        merge_sort(arr, start, mid);
        merge_sort(arr, mid + 1, end);
        merge(arr, start, mid, end);
    }
}

// Parallel Bubble Sort
void parallel_bubble_sort(vector<int>& arr, int n) {
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        #pragma omp parallel for shared(arr, sorted)
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
        #pragma omp parallel for shared(arr, sorted)
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
    }
    for(int i=0;i<n;i++){
        cout<<arr[i]<<" ";
    }
}

// Parallel Merge Sort
void parallel_merge_sort(vector<int>& arr, int start, int end) {
    if (start < end) {
        int mid = start + (end - start) / 2;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_merge_sort(arr, start, mid);
            #pragma omp section
            parallel_merge_sort(arr, mid + 1, end);
        }
        merge(arr, start, mid, end);
    }
}

int main() {
    cout << "Enter number of elements: ";
    int n;
    cin >> n;

    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
        cout<<arr[i]<<" ";
    }
    
    vector<int> arr_copy = arr;
    
    cout<<"\n\nSequential Execution : \n\n";

    cout<<"Bubble Sort : ";
    auto start = high_resolution_clock::now();
    bubble_sort(arr, n);
    auto end = high_resolution_clock::now();
    double seq_bubble_time = duration<double, milli>(end - start).count();
    cout << "\nTIME TAKEN: " << seq_bubble_time << " ms\n";
    
    cout<<"\nMerge Sort : ";
    arr = arr_copy;
    start = high_resolution_clock::now();
    merge_sort(arr, 0, n - 1);
    end = high_resolution_clock::now();
    double seq_merge_time = duration<double, milli>(end - start).count();
    cout << "\nTIME TAKEN: " << seq_merge_time << " ms\n";


    cout<<"\nParallel Execution : \n\n";

    cout<<"Bubble Sort : \n";
    arr = arr_copy;
    start = high_resolution_clock::now();
    parallel_bubble_sort(arr, n);
    end = high_resolution_clock::now();
    double par_bubble_time = duration<double, milli>(end - start).count();
    cout << "TIME TAKEN: " << par_bubble_time << " ms\n";
    cout << "Speedup Factor (Bubble Sort): " << seq_bubble_time / par_bubble_time << "\n";
    
    cout<<"\nMerge Sort : \n";
    arr = arr_copy;
    start = high_resolution_clock::now();
    parallel_merge_sort(arr, 0, n - 1);
    end = high_resolution_clock::now();
    double par_merge_time = duration<double, milli>(end - start).count();
    cout << "TIME TAKEN: " << par_merge_time << " ms\n";
    cout << "Speedup Factor (Merge Sort): " << seq_merge_time / par_merge_time << "\n";
    
    return 0;
}

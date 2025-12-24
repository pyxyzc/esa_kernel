#include <thrust/device_vector.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>
#include <vector>

// Comparator for ZipIterators: Compares only the first element (the value)
struct ValueComparator {
    __host__ __device__
    bool operator()(const thrust::tuple<int, int>& a, const thrust::tuple<int, int>& b) const {
        return thrust::get<0>(a) < thrust::get<0>(b);
    }
};

void filter_unkept_elements(
    thrust::device_vector<int>& keys, 
    thrust::device_vector<int>& old_values, 
    thrust::device_vector<int>& new_values,
    thrust::device_vector<int>& out_keys,
    thrust::device_vector<int>& out_new_values) 
{
    // 1. Sort inputs. Both ranges must be sorted by the comparison value.
    thrust::sort_by_key(new_values.begin(), new_values.end(), keys.begin());
    thrust::sort(old_values.begin(), old_values.end());

    // 2. Create Zip Iterators for the "New" range to keep values and keys together
    auto new_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(new_values.begin(), keys.begin()));
    auto new_zip_end   = thrust::make_zip_iterator(thrust::make_tuple(new_values.end(), keys.end()));

    // Create Zip Iterators for the "Old" range (we only care about the values here)
    // We use a dummy sequence for the second part of the zip to match types
    thrust::device_vector<int> dummy(old_values.size(), 0); 
    auto old_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(old_values.begin(), dummy.begin()));
    auto old_zip_end   = thrust::make_zip_iterator(thrust::make_tuple(old_values.end(), dummy.end()));

    // Output Zip Iterator
    auto out_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(out_new_values.begin(), out_keys.begin()));

    // 3. Perform set_difference using the custom comparator
    auto result_end = thrust::set_difference(
        new_zip_begin, new_zip_end,
        old_zip_begin, old_zip_end,
        out_zip_begin,
        ValueComparator()
    );

    // 4. Resize outputs
    size_t num_unkept = result_end - out_zip_begin;
    out_new_values.resize(num_unkept);
    out_keys.resize(num_unkept);
}

int main() {
    std::vector<int> h_keys = {10, 20, 30, 40, 50, 60};
    std::vector<int> h_new  = {1, 2, 3, 4, 7, 8};
    std::vector<int> h_old  = {2, 3, 4, 1, 9, 10};

    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_new = h_new;
    thrust::device_vector<int> d_old = h_old;

    thrust::device_vector<int> res_keys(h_new.size());
    thrust::device_vector<int> res_vals(h_new.size());

    filter_unkept_elements(d_keys, d_old, d_new, res_keys, res_vals);

    std::cout << "Un-kept Elements (Found in NEW but not in OLD):" << std::endl;
    for(size_t i = 0; i < res_keys.size(); ++i) {
        std::cout << "Value: " << res_vals[i] << " (Key: " << res_keys[i] << ")" << std::endl;
    }

    return 0;
}


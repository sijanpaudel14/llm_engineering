#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <limits>
#include <iomanip>

class LCG {
private:
    uint32_t value;
    const uint32_t a = 1664525;
    const uint32_t c = 1013904223;
    const uint32_t m = 0xFFFFFFFF; // 2^32 - 1

public:
    LCG(uint32_t seed) : value(seed) {}

    uint32_t next() {
        value = (a * value + c) & m; // Using bitwise AND for modulus 2^32
        return value;
    }
};

int64_t max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    LCG lcg_gen(seed);
    std::vector<int> random_numbers;
    random_numbers.reserve(n);
    
    int range = max_val - min_val + 1;
    for (int i = 0; i < n; ++i) {
        uint32_t rand_val = lcg_gen.next();
        int num = min_val + static_cast<int>(rand_val % range);
        random_numbers.push_back(num);
    }

    int64_t max_sum = std::numeric_limits<int64_t>::min();
    for (int i = 0; i < n; ++i) {
        int64_t current_sum = 0;
        for (int j = i; j < n; ++j) {
            current_sum += random_numbers[j];
            if (current_sum > max_sum) {
                max_sum = current_sum;
            }
        }
    }
    return max_sum;
}

int64_t total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    LCG lcg_gen(initial_seed);
    int64_t total_sum = 0;
    
    for (int i = 0; i < 20; ++i) {
        uint32_t seed = lcg_gen.next();
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    
    return total_sum;
}

int main() {
    int n = 10000;
    uint32_t initial_seed = 42;
    int min_val = -10;
    int max_val = 10;

    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << duration << " seconds" << std::endl;
    
    return 0;
}

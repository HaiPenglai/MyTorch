#include <stdio.h>

void _print_tensor(float* tensor, int* shape, int ndim, int left, int right, int shape_pointer) {
    // suppose the tensor is a 4-dim tensor, and the shape is [2, 3, 4, 5], and we want to print the second part which range from [60, 119]
    // so left = 60, right = 120, shape_pointer = 1 (which means the shape of current tensor is [3, 4, 5])
    // num_sub_tensor = 3, nele_sub_tensor = 20, and we can divide it into 3 parts, each part has 20 elements
    // one range from [60, 79], one range from [80, 99], one range from [100, 119]

    int indent_level = shape_pointer;
    int j;

    if (shape_pointer == ndim - 1) {
        printf("[");
        int i;
        for (i = left; i < right; i++) {
            printf("%8.4f", tensor[i]);
            if (i != right - 1) {
                printf(", ");
            }
        }
        printf("]");
    }
    else {
        printf("[");
        int num_sub_tensor = shape[shape_pointer];
        int nele_sub_tensor = (right - left) / num_sub_tensor;
        int i;
        for (i = 0; i < num_sub_tensor; i++) {
            if (i > 0) {
                printf("\n");
                for (j = 0; j <= indent_level; j++) {
                    printf(" ");
                }
            }
            _print_tensor(tensor, shape, ndim, left + i * nele_sub_tensor, left + (i + 1) * nele_sub_tensor, shape_pointer + 1);
            if (i != num_sub_tensor - 1) {
                printf(",");
                if (shape_pointer < ndim - 2) {
                    printf("\n");
                }
            }
        }
        printf("]");
    }
}

void print_tensor(float* tensor, int* shape, int ndim) {
    printf("tensor(");
    int nele = 1;
    int i;
    for (i = 0; i < ndim; i++) {
        nele *= shape[i];
    }
    _print_tensor(tensor, shape, ndim, 0, nele, 0);
    printf(")\n");
}

void test_print_tensor() {
    // we use array to represent tensor, if shape is [2, 3, 4], then the 24 elements are divided into 2 parts, each part has 12 elements
    // then dived 12 elements into 3 parts, each part has 4 elements, which represent 1 vector. and the 3 vectors are combined into 1 matrix.
    // then the 2 matrices are combined into 1 tensor. And if the dim is more than 3, we can continue to divide the tensor into more parts.

    int shape[] = { 2, 3, 4, 5 };
    float tensor[2 * 3 * 4 * 5];
    int i;
    for (i = 0; i < 2 * 3 * 4 * 5; i++) {
        tensor[i] = i;
    }

    print_tensor(tensor, shape, 4);

}

int main() {
    test_print_tensor();
    return 0;
}
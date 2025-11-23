#include <stdio.h>

int simple_tree(float *features, int n_features) {
    if (n_features < 2) return -1;
    if (features[0] > 0) {
        return 0;
    } else {
        if (features[1] > 0) {
            return 0;
        } else {
            return 1;
        }
    }
}

int simple_tree_no_conditions(float *features, int n_features) {
    int c1 = (features[0] <= 0);
    int c2 = (features[1] <= 0);
    return c1 * c2;
}

int main() {
    printf("Testing simple_tree:\n");
    float test1[] = {1.0, -1.0};
    printf("[%f, %f] -> %d (expected: 0)\n", test1[0], test1[1], simple_tree(test1, 2));
    float test2[] = {-1.0, 1.0};
    printf("[%f, %f] -> %d (expected: 0)\n", test2[0], test2[1], simple_tree(test2, 2));
    float test3[] = {-1.0, -1.0};
    printf("[%f, %f] -> %d (expected: 1)\n", test3[0], test3[1], simple_tree(test3, 2));
    
    printf("\nTesting simple_tree_no_conditions:\n");
    printf("[%f, %f] -> %d (expected: 0)\n", test1[0], test1[1], simple_tree_no_conditions(test1, 2));
    printf("[%f, %f] -> %d (expected: 0)\n", test2[0], test2[1], simple_tree_no_conditions(test2, 2));
    printf("[%f, %f] -> %d (expected: 1)\n", test3[0], test3[1], simple_tree_no_conditions(test3, 2));
    return 0;
}

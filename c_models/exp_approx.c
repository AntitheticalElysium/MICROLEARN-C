#include <stdio.h>

float exp_approx(float x, int n_terms) {
    float result = 1.0;
    float term = 1.0;
    for (int i = 1; i <= n_terms; i++) {
        term *= x / i;
        result += term;
    }
    return result;
}

int main() {
    float x = 1.0;
    int n_terms = 10;
    float result = exp_approx(x, n_terms);
    printf("exp_approx(%f, %d) = %f\n", x, n_terms, result);
    printf("Expected: ~2.718282\n");
    return 0;
}

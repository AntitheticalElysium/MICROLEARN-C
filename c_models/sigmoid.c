#include <stdio.h>

float exp_approx(float x, int n_terms) {
    float result = 1.0;
    float term = 1.0;
    for (int i = 1; i < n_terms; i++) {
        term *= x / i;
        result += term;
    }
    return result;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp_approx(-x, 10));
}

int main() {
    float x = 0.0;
    printf("sigmoid(%f) = %f (expected: 0.5)\n", x, sigmoid(x));
    x = 2.0;
    printf("sigmoid(%f) = %f (expected: ~0.88)\n", x, sigmoid(x));
    x = -2.0;
    printf("sigmoid(%f) = %f (expected: ~0.12)\n", x, sigmoid(x));
    return 0;
}

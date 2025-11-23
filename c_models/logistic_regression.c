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

float logistic_regression(float* features, float* thetas, int n_parameters) {
    float z = thetas[0];
    for (int i = 1; i < n_parameters; i++) {
        z += features[i - 1] * thetas[i];
    }
    return sigmoid(z);
}

int main() {
    float features[] = {1.0, 1.0, 1.0};
    float thetas[] = {0.0, 1.0, 1.0, 1.0};
    int n_parameters = 4;
    float result = logistic_regression(features, thetas, n_parameters);
    printf("Logistic Regression Prediction: %f\n", result);
    printf("Expected: ~0.95\n");
    return 0;
}

#include <stdio.h>

float linear_regression_prediction(float* features, float* thetas, int n_parameters) {
    float prediction = thetas[0];
    for (int i = 1; i < n_parameters; i++) {
        prediction += features[i - 1] * thetas[i];
    }
    return prediction;
}

int main() {
    float features[] = {1.0, 1.0, 1.0};
    float thetas[] = {0.0, 1.0, 1.0, 1.0};
    int n_parameters = 4;
    
    float result = linear_regression_prediction(features, thetas, n_parameters);
    printf("Linear Regression Prediction: %f\n", result);
    printf("Expected: 3.0\n");
    return 0;
}

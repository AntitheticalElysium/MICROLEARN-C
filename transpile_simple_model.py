import joblib
import numpy as np
import subprocess
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class CTranspiler:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        print(f"Loaded {type(self.model).__name__}")

    def generate_exp_approx(self):
        return """
float exp_approx(float x, int n_terms) {
    float result = 1.0;
    float term = 1.0;
    for (int i = 1; i < n_terms; i++) {
        term *= x / i;
        result += term;
    }
    return result;
}
"""

    def generate_sigmoid(self):
        return """
float sigmoid(float x) {
    return 1.0 / (1.0 + exp_approx(-x, 10));
}
"""

    def generate_linear_regression(self):
        coef = self.model.coef_
        intercept = self.model.intercept_
        n_features = len(coef)
        coef_str = ", ".join([f"{c}f" for c in coef])

        return f"""
#define N_FEATURES {n_features}
static const float INTERCEPT = {intercept}f;
static const float COEFFICIENTS[N_FEATURES] = {{{coef_str}}};

float prediction(float *features, int n_features) {{
    if (n_features != N_FEATURES) return -1.0f;
    float result = INTERCEPT;
    for (int i = 0; i < N_FEATURES; i++) {{
        result += features[i] * COEFFICIENTS[i];
    }}
    return result;
}}
"""

    def generate_logistic_regression(self):
        coef = (
            self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        )
        intercept = (
            self.model.intercept_[0]
            if hasattr(self.model.intercept_, "__len__")
            else self.model.intercept_
        )
        n_features = len(coef)
        coef_str = ", ".join([f"{c}f" for c in coef])

        return f"""
#define N_FEATURES {n_features}
static const float INTERCEPT = {intercept}f;
static const float COEFFICIENTS[N_FEATURES] = {{{coef_str}}};

float prediction(float *features, int n_features) {{
    if (n_features != N_FEATURES) return -1.0f;
    float z = INTERCEPT;
    for (int i = 0; i < N_FEATURES; i++) {{
        z += features[i] * COEFFICIENTS[i];
    }}
    return sigmoid(z);
}}
"""

    def generate_tree_recursive(self, tree, node_id=0, depth=0):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        indent = "    " * depth

        if left_child == right_child:
            value = tree.value[node_id]
            class_prediction = (
                np.argmax(value[0]) if len(value.shape) > 1 else int(value[0])
            )
            return f"{indent}return {class_prediction};\n"

        code = f"{indent}if (features[{feature}] <= {threshold}f) {{\n"
        code += self.generate_tree_recursive(tree, left_child, depth + 1)
        code += f"{indent}}} else {{\n"
        code += self.generate_tree_recursive(tree, right_child, depth + 1)
        code += f"{indent}}}\n"
        return code

    def generate_decision_tree(self):
        tree = self.model.tree_
        n_features = len(tree.feature)

        return f"""
#define N_FEATURES {n_features}

int prediction(float *features, int n_features) {{
    if (n_features != N_FEATURES) return -1;
{self.generate_tree_recursive(tree)}
}}
"""

    def generate_main(self, test_data):
        test_data_str = ", ".join([f"{x}f" for x in test_data])
        n_features = len(test_data)

        return f"""
int main() {{
    float test_features[{n_features}] = {{{test_data_str}}};
    float result = prediction(test_features, {n_features});
    
    printf("Input features: [");
    for (int i = 0; i < {n_features}; i++) {{
        printf("%f", test_features[i]);
        if (i < {n_features} - 1) printf(", ");
    }}
    printf("]\\n");
    printf("Prediction: %f\\n", result);
    
    return 0;
}}
"""

    def transpile(self, output_file="model_inference.c", test_data=None, compile=True):
        code = "#include <stdio.h>\n\n"

        if isinstance(self.model, LogisticRegression):
            code += self.generate_exp_approx()
            code += self.generate_sigmoid()
            code += self.generate_logistic_regression()
        elif isinstance(self.model, LinearRegression):
            code += self.generate_linear_regression()
        elif isinstance(self.model, DecisionTreeClassifier):
            code += self.generate_decision_tree()
        else:
            raise ValueError(f"Unsupported model type: {type(self.model).__name__}")

        if test_data is None:
            if isinstance(self.model, (LinearRegression, LogisticRegression)):
                coef = self.model.coef_
                n_features = coef.shape[1] if len(coef.shape) > 1 else len(coef)
            elif isinstance(self.model, DecisionTreeClassifier):
                n_features = len(self.model.tree_.feature)
            else:
                n_features = 3
            test_data = np.ones(n_features)

        code += self.generate_main(test_data)

        with open(output_file, "w") as f:
            f.write(code)

        print(f"Generated: {output_file}")

        if compile:
            output_binary = output_file.replace(".c", "")
            cmd = f"gcc {output_file} -o {output_binary} -lm -O2"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Compiled: {output_binary}")
            else:
                print(f"Compilation error: {result.stderr}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transpile_simple_model.py <model.joblib> [output.c]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "model_inference.c"

    transpiler = CTranspiler(model_path)
    transpiler.transpile(output_file=output_file, compile=True)

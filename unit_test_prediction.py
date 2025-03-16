import joblib
import numpy as np
import logging
import sklearn
import unittest

# Set up logging
logging.basicConfig(
    filename="model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Check scikit-learn version compatibility
expected_version = "1.4.1.post1"  # The version used to train the model
current_version = sklearn.__version__

if current_version != expected_version:
    logging.warning(f"⚠️ Warning: Model was trained with scikit-learn {expected_version}, but you are using {current_version}.")

# Load the trained model
try:
    model = joblib.load("best_model.pkl")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None  # Ensure model is None if loading fails

def predict(input_features):
    """
    Takes a list of 23 numerical features and predicts the target value.
    """
    if model is None:
        logging.error("❌ Prediction failed: Model not loaded.")
        return None

    try:
        # Ensure input is a NumPy array with the correct shape
        input_array = np.array(input_features).reshape(1, -1)

        # Check if input shape matches model expectation
        expected_features = 23
        if input_array.shape[1] != expected_features:
            logging.error(f"❌ Prediction error: X has {input_array.shape[1]} features, but model expects {expected_features} features.")
            return None

        # Make prediction
        prediction = model.predict(input_array)

        # Log successful prediction
        logging.info(f"✅ Prediction successful: {prediction[0]}")
        return prediction[0]  # Return single value
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        return None

# ✅ Example input (Ensure it has exactly 23 features)
if __name__ == "__main__":
    test_input = [0.41, 0.47, 0.46, 0.37, 0.46, 0.43, 0.43, 0.35, 0.28, 0.31,
                  0.42, 0.39, 0.45, 0.48, 0.29, 0.33, 0.38, 0.44, 0.41, 0.36,
                  0.40, 0.37, 0.35]  # 23 values

    print(f"✅ Input Shape: {len(test_input)} features")  # Debugging step
    predicted_value = predict(test_input)
    print("Predicted Value:", predicted_value)

    # Unit Tests
    class TestPredictionModel(unittest.TestCase):

        def test_valid_input(self):
            """Test prediction with valid input of 23 numerical features"""
            test_input = [0.41, 0.47, 0.46, 0.37, 0.46, 0.43, 0.43, 0.35, 0.28, 0.31,
                          0.42, 0.39, 0.45, 0.48, 0.29, 0.33, 0.38, 0.44, 0.41, 0.36,
                          0.40, 0.37, 0.35]
            result = predict(test_input)
            self.assertIsNotNone(result, "Prediction should return a valid result")

        def test_invalid_input_length(self):
            """Test prediction with incorrect number of features"""
            test_input = [0.41, 0.47, 0.46]  # Only 3 features instead of 23
            result = predict(test_input)
            self.assertIsNone(result, "Prediction should fail due to incorrect input size")

        def test_invalid_input_type(self):
            """Test prediction with non-numeric input"""
            test_input = ["a", "b", "c"] + [0.3] * 20  # Mix of strings and numbers
            result = predict(test_input)
            self.assertIsNone(result, "Prediction should fail due to invalid input type")

    # Run tests if this script is executed directly
    unittest.main()

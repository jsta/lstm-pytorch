"""
Generate data following:
https://www.jessicayung.com/generating-autoregressive-data-for-experiments/
"""
import pickle
import importlib

ARData = importlib.import_module("99_generate_data_utils").ARData

# Data params
num_datapoints = 100
test_size = 0.2
noise_var = 0

# Network params
input_size = 20

# A set of coefficients that are stable (to produce replicable plots, experiments)
fixed_ar_coefficients = {
    2: [0.46152873, -0.29890739],
    5: [0.02519834, -0.24396899, 0.2785921, 0.14682383, 0.39390468],
    10: [
        -0.10958935,
        -0.34564819,
        0.3682048,
        0.3134046,
        -0.21553732,
        0.34613629,
        0.41916508,
        0.0165352,
        0.14163503,
        -0.38844378,
    ],
    20: [
        0.1937815,
        0.01201026,
        0.00464018,
        -0.21887467,
        -0.20113385,
        -0.02322278,
        0.34285319,
        -0.21069086,
        0.06604683,
        -0.22377364,
        0.11714593,
        -0.07122126,
        -0.16346554,
        0.03174824,
        0.308584,
        0.06881604,
        0.24840789,
        -0.32735569,
        0.21939492,
        0.3996207,
    ],
}

data = ARData(
    num_datapoints,
    num_prev=input_size,
    test_size=test_size,
    noise_var=noise_var,
    coeffs=fixed_ar_coefficients[input_size],
)

pickle.dump(data, open("../data/data.ardata", "wb"))

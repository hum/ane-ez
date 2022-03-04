#!/usr/bin/env python3
import coremltools as cmt
import numpy as np

from coremltools.models.neural_network import datatypes, NeuralNetworkBuilder

input_name = "input"
output_name = "output"

in_features = [(input_name, datatypes.Array(3))]
out_features = [(output_name, datatypes.Array(2))]

weights = np.zeros((3, 2))
bias = np.ones((2))

builder = NeuralNetworkBuilder(in_features, out_features)
builder.add_inner_product(
    name="in_layer",
    b=bias,
    has_bias=True,
    input_channels=3,
    input_name=input_name,
    output_channels=2,
    output_name=output_name,
    W=weights,
)

model = cmt.models.MLModel(builder.spec)
result = model.predict({"input": np.zeros((3))})
print(result)

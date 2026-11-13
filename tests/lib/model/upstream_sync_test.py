"""CPU regressions for the fork's pinned upstream serialization and encoder fixes.

These tests use synthetic tensors and randomly initialized local models. They do
not load image datasets, download pretrained weights, train models, or publish.
"""

import operator

import cv2
import numpy as np
import pytest
from keras import Input, Model, applications, device, models, ops
from numpy.testing import assert_allclose

from lib.model.layers import ScalarOp
from plugins.train.model import phaze_a


@pytest.mark.parametrize("operation", ["multiply", "truediv", "add", "subtract"])
@pytest.mark.parametrize("callable_operation", [False, True])
def test_scalar_op_restores_serialized_dtype_and_operation(operation, callable_operation):
    """Keras 3 dtype dictionaries and legacy callable operations both restore."""
    with device("cpu"):
        original = ScalarOp(operation=operation, value=2.0, dtype="float32")
        config = original.get_config()
        assert isinstance(config["dtype"], dict)
        if callable_operation:
            config["operation"] = {"multiply": np.multiply, "truediv": operator.truediv,
                                   "add": np.add, "subtract": np.subtract}[operation]
        restored = ScalarOp.from_config(config)
        values = np.array([[1.0, 4.0]], dtype="float32")
        assert_allclose(ops.convert_to_numpy(restored(values)),
                        ops.convert_to_numpy(original(values)))
        assert restored.dtype_policy.name == "float32"


@pytest.mark.parametrize("operation", ["multiply", "truediv", "add", "subtract"])
def test_scalar_op_model_round_trip(tmp_path, operation):
    """A persisted Keras model restores its ScalarOp and numerical behavior."""
    with device("cpu"):
        inputs = Input(shape=(2,))
        model = Model(inputs, ScalarOp(operation=operation, value=2.0)(inputs))
        values = np.array([[1.0, 4.0]], dtype="float32")
        before = ops.convert_to_numpy(model(values))
        path = tmp_path / "scalar.keras"
        model.save(path)
        restored = models.load_model(path, compile=False)
        assert_allclose(ops.convert_to_numpy(restored(values)), before)


@pytest.mark.parametrize("architecture,constructor", [
    ("mobilenet_v3_small", applications.MobileNetV3Small),
    ("mobilenet_v3_large", applications.MobileNetV3Large),
])
def test_phaze_mobilenet_v3_kwargs_build_real_encoder(monkeypatch, architecture, constructor):
    """Both Phaze-A keys supply valid Keras kwargs without pretrained downloads."""
    for name, value in {"mobilenet_width": 1.0, "mobilenet_depth": 1,
                        "mobilenet_dropout": 0.0, "mobilenet_minimalistic": True}.items():
        monkeypatch.setattr(phaze_a.cfg, name, lambda result=value: result)
    encoder = phaze_a.Encoder((32, 32, 3))
    kwargs = encoder._model_kwargs[architecture]  # pylint:disable=protected-access
    assert kwargs["minimalistic"] is True
    assert kwargs["include_preprocessing"] is False
    assert kwargs["alpha"] == 1.0
    with device("cpu"):
        model = constructor(input_shape=(32, 32, 3), weights=None, include_top=False, **kwargs)
        result = ops.convert_to_numpy(model(np.zeros((1, 32, 32, 3), dtype="float32")))
        assert result.shape[0] == 1
        assert np.isfinite(result).all()


def test_legacy_caffe_cpu_dnn_inference(tmp_path):
    """Keep the real Caffe importer required by the existing cv2-dnn detector."""
    prototxt = tmp_path / "relu.prototxt"
    prototxt.write_text(
        'name: "cpu_compat"\ninput: "data"\n'
        'input_shape { dim: 1 dim: 3 dim: 1 dim: 1 }\n'
        'layer { name: "relu" type: "ReLU" bottom: "data" top: "relu" }\n',
        encoding="utf-8",
    )
    model = cv2.dnn.readNetFromCaffe(str(prototxt))
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    values = np.array([-1.0, 0.0, 1.0], dtype="float32").reshape(1, 3, 1, 1)
    model.setInput(values)
    assert_allclose(model.forward(), np.maximum(values, 0.0))

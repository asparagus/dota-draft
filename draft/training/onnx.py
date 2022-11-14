"""Script to try out an onnx model."""
import argparse
import numpy as np
import onnxruntime as ort


INPUT_NAME = 'processed.1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Try an exported onnx model.')
    parser.add_argument('onnx_path', help='Path to the saved onnx model')
    args = parser.parse_args()

    radiant = [1, 2, 3, 4, 5]
    dire = [6, 7, 8, 9, 10]
    sample_input = radiant + dire

    ort_session = ort.InferenceSession(args.onnx_path)
    outputs = ort_session.run(
        None, 
        {
            INPUT_NAME: np.array([sample_input]).astype(np.int32)
        },
    )
    print(outputs)

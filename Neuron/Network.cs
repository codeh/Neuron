using System;
using Neuron.Layers;

namespace Neuron
{
    public class Network
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly Layer[] _layers;
        private readonly ErrorCalculator _errorCalculator;

        public Network(int inputSize, int outputSize, Layer[] layers, ErrorCalculator errorCalculator)
        {
            if (inputSize <= 0) throw new ArgumentException($"Input size should be greater then 0");
            if (outputSize <= 0) throw new ArgumentException($"Output size should be greater then 0");
            
            _inputSize = inputSize;
            _outputSize = outputSize;
            _layers = layers ?? throw new ArgumentNullException(nameof(layers));
            _errorCalculator = errorCalculator ?? throw new ArgumentNullException(nameof(errorCalculator));
            
            CheckLayers();
        }

        private void CheckLayers()
        {
            var inputSize = _inputSize;

            foreach (var layer in _layers)
            {
                if (layer.InputSize != inputSize) throw new Exception("Invalid layer size");
                inputSize = layer.LayerSize;
            }
            
            if (inputSize != _outputSize) throw new Exception("Invalid layer size");
        }

        public void Learn(double[][] inputs, double[][] targets, int epochs)
        {
            for (var e = 0; e < epochs; e++)
            {
                var errorSumm = 0.0;
                for (var i = 0; i < inputs.Length; i++)
                {
                    double[] input = inputs[i];
                    foreach (var layer in _layers)
                    {
                        input = layer.GetOutput(input);
                    }

                    var error = _errorCalculator.CalcError(input, targets[i]);
                    errorSumm += error;

                    input = _errorCalculator.GetDerivatives(input, targets[i]);
                    for (var l = _layers.Length - 1; l >= 0; l--)
                    {
                        input = _layers[l].BackPropagate(input);
                    }
                }
                Console.WriteLine($"Epoch {e} average error {errorSumm / inputs.Length}");
            }
        }

        public double[] GetOutput(double[] input)
        {
            var array = input;
            foreach (var layer in _layers)
            {
                array = layer.GetOutput(array);
            }

            return array;
        }
    }
}
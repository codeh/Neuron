using System;

namespace Neuron.Layers
{
    public abstract class Layer
    {
        private readonly double _learnSpeed;
        public int InputSize { get; }
        public int LayerSize { get; }
        protected double[,] Weights { get; }
        protected double[] Input { get; set; }
        protected double[] Output { get; }
        protected double[] Sigma { get; }
        protected double[] SigmaWeightedSumms { get; }

        protected Layer(int inputSize, int layerSize, double learnSpeed = 0.1)
        {
            _learnSpeed = learnSpeed;
            InputSize = inputSize;
            LayerSize = layerSize;
            Weights = new double[inputSize, layerSize];
            Output = new double[layerSize];
            Sigma = new double[layerSize];
            SigmaWeightedSumms = new double[inputSize];
            
            var rnd = new Random();
            for (var i = 0; i < inputSize; i++)
            {
                for (var j = 0; j < layerSize; j++)
                {
                    Weights[i, j] = rnd.NextDouble() * 0.5;
                }
            }
        }

        public double[] GetOutput(double[] input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Length != InputSize) throw new ArgumentException($"Input size {InputSize} expected, but {input.Length} received");

            Input = input;
            
            for (var j = 0; j < LayerSize; j++)
            {
                var summ = 0.0;
                for (var i = 0; i < InputSize; i++)
                {
                    summ += input[i] * Weights[i, j];
                }

                Output[j] = OutFunc(summ);
            }

            return Output;
        }

        private double[] GetSigmaWeightSumms()
        {
            for (var i = 0; i < InputSize; i++)
            {
                var summ = 0.0;
                for (var j = 0; j < LayerSize; j++)
                {
                    summ += Sigma[j] * Weights[i, j];
                }

                SigmaWeightedSumms[i] = summ;
            }

            return SigmaWeightedSumms;
        }

        public double[] BackPropagate(double[] sigmaWeightSumms)
        {
            if (sigmaWeightSumms == null) throw new ArgumentNullException(nameof(sigmaWeightSumms));
            if (sigmaWeightSumms.Length != LayerSize) throw new ArgumentException($"sigmaWeightSumms size {LayerSize} expected, but {sigmaWeightSumms.Length} received");

            for (var i = 0; i < LayerSize; i++)
            {
                Sigma[i] = OutFuncDerivative(Output[i]) * sigmaWeightSumms[i];
            }
            
            var sigmaWeights = GetSigmaWeightSumms();
            
            AdjustWeights();

            return sigmaWeights;
        }

        private void AdjustWeights()
        {
            for (var i = 0; i < InputSize; i++)
            {
                for (var j = 0; j < LayerSize; j++)
                {
                    Weights[i, j] -= _learnSpeed * Sigma[j] * Input[i];
                }
            }
        }

        protected abstract double OutFunc(double val);
        protected abstract double OutFuncDerivative(double val);
    }
}
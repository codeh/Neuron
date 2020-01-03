using System;

namespace Neuron.Layers
{
    public class SigmoidLayer : Layer
    {
        private readonly double _alpha;

        public SigmoidLayer(int inputSize, int layerSize, double alpha = 1.0, double learnSpeed = 0.1) 
            : base(inputSize, layerSize, learnSpeed)
        {
            _alpha = alpha;
        }

        protected override double OutFunc(double val)
        {
            return 1.0 / (1.0 + Math.Exp(-2.0 * _alpha * val));
        }

        protected override double OutFuncDerivative(double val)
        {
            return 2 * _alpha * val * (1 - val);
        }
    }
}
using System;

namespace Neuron
{
    public abstract class ErrorCalculator
    {
        public double CalcError(double[] output, double[] target)
        {
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (output.Length == 0) throw new ArgumentException("Output length should be greater then 0");
            if (output.Length != target.Length) throw new ArgumentException($"Output length is {output.Length}, but target length is {target.Length}. They should be equal");

            return ErrorFunc(output, target);
        }

        public double[] GetDerivatives(double[] output, double[] target)
        {
            var derivatives = new double[output.Length];
            for (var i = 0; i < output.Length; i++)
            {
                derivatives[i] = ErrorFuncDerivative(output[i], target[i]);
            }

            return derivatives;
        }

        protected abstract double ErrorFunc(double[] output, double[] target);
        protected abstract double ErrorFuncDerivative(double output, double target);
    }
}
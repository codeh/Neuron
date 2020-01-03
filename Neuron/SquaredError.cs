namespace Neuron
{
    public class SquaredError : ErrorCalculator
    {
        protected override double ErrorFunc(double[] output, double[] target)
        {
            var summ = 0.0;
            for (var i = 0; i < output.Length; i++)
            {
                summ += (target[i] - output[i]) * (target[i] - output[i]);
            }

            return summ * 0.5;
        }

        protected override double ErrorFuncDerivative(double output, double target)
        {
            return output - target;
        }
    }
}
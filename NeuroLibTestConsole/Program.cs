using System;
using Neuron;
using Neuron.Layers;

namespace NeuroLibTestConsole
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            var layers = new Layer[]
            {
                new SigmoidLayer(2, 4),
                new SigmoidLayer(4, 1) 
            };
            
            var errorCalculator = new SquaredError();
            
            var network = new Network(2, 1, layers, errorCalculator);

            var size = 10000;
            
            var inputs = new double[size][];
            var target = new double[size][];

            var rnd = new Random();
            for (var i = 0; i < size; i++)
            {
                var x = rnd.NextDouble();
                var y = rnd.NextDouble();
                
                inputs[i] = new[]{x, y};
                target[i] = new[]{x * y};
            }
            
            network.Learn(inputs, target, 1000);

            for (var i = 0; i < 10; i++)
            {
                var x = rnd.NextDouble();
                var y = rnd.NextDouble();
                
                Console.WriteLine($"{x:F}*{y:F} = {x*y:F}, network says {network.GetOutput(new[]{x,y})[0]:F}");
            }
        }
    }
}
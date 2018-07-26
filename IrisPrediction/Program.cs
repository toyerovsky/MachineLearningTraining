using System;
using System.IO;
using IrisPrediction.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace IrisPrediction
{
    class Program
    {
        internal static string DataPath = Path.Combine(Environment.CurrentDirectory, "iris.data");

        static void Main(string[] args)
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(DataPath).CreateFrom<IrisData>(separator: ','));
            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            var model = pipeline.Train<IrisData, Model.IrisPrediction>();

            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.ReadLine();
        }
    }
}

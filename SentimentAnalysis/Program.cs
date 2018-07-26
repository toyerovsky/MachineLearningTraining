using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using SentimentAnalysis.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace SentimentAnalysis
{
    class Program
    {
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wiki-line-data.tsv");
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wiki-line-test.tsv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            var model = await Train();
            Evaluate(model);
            IEnumerable<SentimentData> sentiments = GetDataFromUser();
            Predict(sentiments, model);
            Console.ReadLine();
        }

        public async static Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier()
            {
                NumLeaves = 5,
                NumTrees = 5,
                MinDocumentsInLeafs = 2
            });
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();
            await model.WriteAsync(path: _modelPath);
            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public static void Predict(IEnumerable<SentimentData> sentiments, PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();
        }

        public static IEnumerable<SentimentData> GetDataFromUser()
        {
            var predictions = new HashSet<SentimentData>();
            Console.WriteLine("Type comments to analyze. Press enter to get to the next comment. " +
                "When you're done type 'end'");

            string buffer = Console.ReadLine();
            while (buffer != "end")
            {
                predictions.Add(new SentimentData()
                {
                    SentimentText = buffer,
                });
                buffer = Console.ReadLine();
            }
            return predictions;
        }
    }
}
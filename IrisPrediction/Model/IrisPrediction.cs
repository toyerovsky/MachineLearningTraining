using Microsoft.ML.Runtime.Api;

namespace IrisPrediction.Model
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
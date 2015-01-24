using MlLight.Classifiers;
using MlLight.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MlLight.Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            //Naive Bayes multinomial classifier demo
            
            //init sample instances
            List<Instance> trainInstances = new List<Instance>();
            trainInstances.Add(new Instance() { Id = 1, Tokens = new List<string>() { "Chinese", "Beijing", "Chinese" }, Category = "Yes" });
            trainInstances.Add(new Instance() { Id = 2, Tokens = new List<string>() { "Chinese", "Chinese", "Shanghai" }, Category = "Yes" });
            trainInstances.Add(new Instance() { Id = 3, Tokens = new List<string>() { "Chinese", "Macao" }, Category = "Yes" });
            trainInstances.Add(new Instance() { Id = 4, Tokens = new List<string>() { "Chinese", "Tokyo", "Japan" }, Category = "no" });

            Instance testInstance = new Instance() { Id = 5, Tokens = new List<string>() { "Chinese", "Chinese", "Tokyo", "Japan" }, Category = "" };

            //train
            NaiveBayesMultinomialClassifier classifier = new NaiveBayesMultinomialClassifier();
            classifier.Train(trainInstances);

            //classify
            string category = classifier.Classify(testInstance);
        }
    }
}

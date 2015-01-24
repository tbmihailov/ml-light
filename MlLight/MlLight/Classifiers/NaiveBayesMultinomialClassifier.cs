using MlLight.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MlLight.Classifiers
{
    /// <summary>
    /// NaiveBayes Multinomial Classifier
    /// A good explanation of the NB multinomial classifier is located here http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
    /// </summary>
    public class NaiveBayesMultinomialClassifier
    {
        Dictionary<string, double> _categoryProbabilities;//prior probabilities
        Dictionary<string, double> _termCategoryProbabilities;//conditional probabilities
        List<string> _categories;//all categories in the classifier
        List<string> _termsVocabulary;//term vocabulary of all terms

        /// <summary>
        /// Train classifier with instances
        /// </summary>
        /// <param name="instances">Train instances</param>
        public void Train(List<Instance> instances)
        {
            _categoryProbabilities = new Dictionary<string, double>();
            _termCategoryProbabilities = new Dictionary<string, double>();
            _termsVocabulary = ExtractVocabulary(instances);
            _categories = ExtractClasses(instances);

            int allDocumentsCount = instances.Count;
            foreach (var category in _categories)
            {
                var documentsInCategory = instances.Where(ti => ti.Category == category);
                int documentsInCategoryCount = documentsInCategory.Count();

                _categoryProbabilities[category] = documentsInCategoryCount / allDocumentsCount;

                int allTokensInCategory = GetAllTokensCount(documentsInCategory);
                foreach (var term in _termsVocabulary)
                {
                    int numberOfTokensOfTermInCategory = GetNumberOfTokensOfTerm(documentsInCategory, term);
                    double conditionalProbabilityOfTermInCategory = numberOfTokensOfTermInCategory + 1 / (allTokensInCategory + _termsVocabulary.Count);
                    _termCategoryProbabilities[GetTermInCategoryKey(category, term)] = conditionalProbabilityOfTermInCategory;
                }
            }
        }

        /// <summary>
        /// Returns key in the conditional probability dictionary
        /// </summary>
        /// <param name="category">Category in which is the term probability calculated</param>
        /// <param name="term">Term</param>
        /// <returns></returns>
        private static string GetTermInCategoryKey(string category, string term)
        {
            return "t_" + term + "_c_" + category;
        }

        /// <summary>
        /// Counts all tokens in a set of documents
        /// </summary>
        /// <param name="instances">Set of documents</param>
        /// <returns>Number of tokens in a set of documents</returns>
        private int GetAllTokensCount(IEnumerable<Instance> instances)
        {
            return instances.Sum(d => d.Tokens.Count);
        }

        /// <summary>
        /// Counts the number of tokens of a given term  in set of documents
        /// </summary>
        /// <param name="instances">Documents in set</param>
        /// <param name="term">Term to look for</param>
        /// <returns>Number of tokens of a given term  in set of documents</returns>
        private int GetNumberOfTokensOfTerm(IEnumerable<Instance> instances, string term)
        {
            return instances.Sum(d => d.Tokens.Count(t => t == term));
        }

        /// <summary>
        /// Counts the number documents of a given category in a set of documents
        /// </summary>
        /// <param name="instances">Set of documents</param>
        /// <param name="category">Category for count docs for</param>
        /// <returns>Number documents of a given category in a set of documents</returns>
        private int CountDocumentsInCategory(List<Instance> instances, string category)
        {
            return instances.Count(d => d.Category == category);
        }

        /// <summary>
        /// Extract classes from a set of documents
        /// </summary>
        /// <param name="instances">Set of documents for extract classes from</param>
        /// <returns>Classes that appear in a set of documents</returns>
        private List<string> ExtractClasses(List<Instance> instances)
        {
            return instances.Select(d => d.Category).Distinct().ToList();
        }

        /// <summary>
        /// Extract term vocabulary from a set of documents
        /// </summary>
        /// <param name="trainInstances">Set of documents to extract vocabularies from</param>
        /// <returns>Vocabulary of distinct terms found in a document set</returns>
        private List<string> ExtractVocabulary(List<Instance> trainInstances)
        {
            return trainInstances.SelectMany(d => d.Tokens).Distinct().ToList();
        }

        /// <summary>
        /// Classify
        /// </summary>
        /// <param name="instance"></param>
        /// <returns></returns>
        public string Classify(Instance instance)
        {
            Dictionary<string, double> rankedCategories = GetCategoriesWithScoresForDocument(instance);
            string category = rankedCategories.First(rc => rc.Value == rankedCategories.Max(rc1 => rc1.Value)).Key;

            return category;
        }

        /// <summary>
        /// Rank every category in the classifier to the given instance to be classified
        /// </summary>
        /// <param name="instance">Instance to be classified</param>
        /// <returns>Dictionary of categories ranked to a given document</returns>
        private Dictionary<string, double> GetCategoriesWithScoresForDocument(Instance instance)
        {
            Dictionary<string, double> categoriesWithRank = new Dictionary<string, double>();
            List<string> instanceTokens = instance.Tokens;

            foreach (var category in _categories)
            {
                categoriesWithRank[category] = Math.Log(_categoryProbabilities[category], 2);
                foreach (var token in instanceTokens)
                {
                    if (!_termsVocabulary.Contains(token))
                    {
                        continue;
                    }

                    categoriesWithRank[category] += Math.Log(_termCategoryProbabilities[GetTermInCategoryKey(category, token)], 2);
                }
            }

            return categoriesWithRank;
        }
    }
}

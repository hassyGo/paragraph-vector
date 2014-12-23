#include "Matrix.hpp"
#include <unordered_map>
#include <vector>

typedef unsigned int INDEX;
typedef unsigned long int COUNT;

class Vocabulary{
public:
  Vocabulary(const int wordVectorDim, const int contextLength, const int paragraphVectorDim);

  std::unordered_map<std::string, INDEX> wordIndex;
  std::vector<std::string> wordList;
  std::vector<COUNT> wordCount;

  const int contextLen;
  const int wordVecDim;
  const int paragraphVecDim;
  const int wordScoreVecDim;
  INDEX unkIndex;
  INDEX nullIndex;

  MatD wordVector; //a column vector for each word
  MatD paragraphVector;
  MatD wordScoreVector;

  std::vector<INDEX> noiseDistribution;
  std::vector<double> discardProb;

  void read(const std::string& documentFile, COUNT freqThreshold);
  void train(const std::string& documentFile, double& learningRate, const double shirnk, const int numNegative);
  void outputParagraphVector(const std::string& fileName);
  void outputWordVector(const std::string& fileName);
  void save(const std::string& fileName);
  void load(const std::string& fileName);
  void wordKnn(const int k);

private:
  void train(const INDEX paragraphIndex, const std::vector<INDEX>& paragraph, const double learningRate, const int numNegative);
};

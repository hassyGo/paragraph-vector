#include "Vocabulary.hpp"
#include "Utils.hpp"

int main(int argc, char** argv){
  int wordVecDim = 50;
  int paragraphVecDim = 50;
  int contextSize = 5;
  double learningRate = 0.025;
  int numNegative = 5;
  int minFreq = 10;
  int iteration = 1;
  int numThreads = 1;
  double shrink = 0.0;
  std::string input = "INPUT.txt";
  std::string output = "OUTPUT";

  Utils::procArg(argc, argv,
		 wordVecDim, paragraphVecDim, contextSize, learningRate, numNegative, minFreq, iteration, numThreads,
		 input, output);

  Vocabulary voc(wordVecDim, contextSize, paragraphVecDim);

  voc.read(input, minFreq);
  shrink = learningRate/iteration;

  for (int i = 0; i < iteration; ++i){
    printf("Iteration %2d (current learning rate: %f)\n", i+1, learningRate);
    voc.train(input, learningRate, shrink, numNegative, numThreads);
    learningRate -= shrink;
    voc.save(output+".bin");
  }

  //voc.wordKnn(10);
  voc.outputParagraphVector(output+".pv");
  voc.outputWordVector(output+".wv");

  return 0;
}

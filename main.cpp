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
  std::string input = "INPUT.txt";
  std::string output = "OUTPUT";

  Utils::procArg(argc, argv,
		 wordVecDim, paragraphVecDim, contextSize, learningRate, numNegative, minFreq, iteration,
		 input, output);

  Vocabulary voc(wordVecDim, contextSize, paragraphVecDim);

  voc.read(input, minFreq);

  for (int i = 0; i < iteration; ++i){
    printf("Iteration %2d\n", i+1);
    voc.train(input, learningRate, numNegative);
    voc.save(output+".bin");
  }

  voc.outputParagraphVector(output+".pv");
  voc.outputWordVector(output+".wv");

  return 0;
}

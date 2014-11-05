#pragma once

#include "Matrix.hpp"

namespace Utils{
  inline bool isSpace(const char& c){
    return (c == ' ' || c == '\t');
  }
  
  inline void infNan(const double& x){
    assert(!isnan(x) && !isinf(x));
  }

  inline unsigned long xor128(){
    static unsigned long x = 123456789, y = 362436069, z = 521288629, w = 88675123; 
    unsigned long t;
    
    t=(x^(x<<11));
    x=y; y=z; z=w;
    return (w=(w^(w>>19))^(t^(t>>8)));
  }

  inline double uniformRandom(){
    return (Utils::xor128()&0xFFFF)/65536.0;
  }
  
  inline double cosDis(const MatD& a, const MatD& b){
    return a.col(0).dot(b.col(0))/(a.norm()*b.norm());
  }

  inline double sigmoid(const double& x){
    return 1.0/(1.0+::exp(-x));
  }

  inline void procArg(int argc, char** argv, int& wordVecDim, int& paragraphVecDim, int& contextSize, double& learningRate, int& numNegative, int& minFreq, int& iteration, std::string& input, std::string& output){
    for (int i = 1; i < argc-1; i+=2){
      std::string arg = (std::string)argv[i];

      if (arg == "-wvdim"){
	wordVecDim = atoi(argv[i+1]);
	assert(wordVecDim > 0);
      }
      else if (arg == "-pvdim"){
	paragraphVecDim = atoi(argv[i+1]);
	assert(paragraphVecDim > 0);
      }
      else if (arg == "-window"){
	contextSize = atoi(argv[i+1]);
	assert(contextSize > 0);
      }
      else if (arg == "-lr"){
	learningRate = atof(argv[i+1]);
	assert(learningRate > 0.0);
      }
      else if (arg == "-neg"){
	numNegative = atoi(argv[i+1]);
	assert(numNegative > 0);
      }
      else if (arg == "-minfreq"){
	minFreq = atoi(argv[i+1]);
	assert(minFreq > 0);
      }
      else if (arg == "-itr"){
	iteration = atoi(argv[i+1]);
	assert(minFreq > 0);
      }
      else if (arg == "-input"){
	input = (std::string)argv[i+1];
      }
      else if (arg == "-output"){
	output = (std::string)argv[i+1];
      }
    }
  }
};

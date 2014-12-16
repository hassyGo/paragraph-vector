#pragma once

#include "Matrix.hpp"
#include <fstream>

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

  inline void save(std::ofstream& ofs, const MatD& mat){
    static double val;

    for (int i = 0; i < mat.cols(); ++i){
      for (int j = 0; j < mat.rows(); ++j){
	val = mat.coeff(j, i);
	Utils::infNan(val);
	ofs.write((char*)&val, sizeof(double));
      }
    }
  }

  inline void load(std::ifstream& ifs, MatD& mat){
    static double val;

    for (int i = 0; i < mat.cols(); ++i){
      for (int j = 0; j < mat.rows(); ++j){
	ifs.read((char*)&val, sizeof(double));
	mat.coeffRef(j, i) = val;
	Utils::infNan(val);
      }
    }
  }
  
  inline void procArg(int argc, char** argv, int& wordVecDim, int& paragraphVecDim, int& contextSize, double& learningRate, int& numNegative, int& minFreq, int& iteration, std::string& input, std::string& output){
    for (int i = 1; i < argc; i+=2){
      std::string arg = (std::string)argv[i];

      if (arg == "-help"){
	printf("### Options ###\n");
	printf("-wvdim    the dimensionality of word vectors (default: 50)\n");
	printf("-pvdim    the dimensionality of paragraph vectors (default: 50)\n");
	printf("-window   the context window size (default: 5)\n");
	printf("-lr       the learning rate (default: 0.025)\n");
	printf("-neg      the number of negative samples for negative sampling learning (default: 5)\n");
	printf("-minfreq  the threshold to cut rare words (default: 10)\n");
	printf("-itr      the number of iterations (default: 1)\n");
	printf("-input    the input file name (default: INPUT.txt)\n");
	printf("-output   the output file name (default: OUTPUT)\n");
	exit(1);
      }
      else if (arg == "-wvdim"){
	assert(i+1 < argc);
	wordVecDim = atoi(argv[i+1]);
	assert(wordVecDim > 0);
      }
      else if (arg == "-pvdim"){
	assert(i+1 < argc);
	paragraphVecDim = atoi(argv[i+1]);
	assert(paragraphVecDim > 0);
      }
      else if (arg == "-window"){
	assert(i+1 < argc);
	contextSize = atoi(argv[i+1]);
	assert(contextSize > 0);
      }
      else if (arg == "-lr"){
	assert(i+1 < argc);
	learningRate = atof(argv[i+1]);
	assert(learningRate > 0.0);
      }
      else if (arg == "-neg"){
	assert(i+1 < argc);
	numNegative = atoi(argv[i+1]);
	assert(numNegative > 0);
      }
      else if (arg == "-minfreq"){
	assert(i+1 < argc);
	minFreq = atoi(argv[i+1]);
	assert(minFreq > 0);
      }
      else if (arg == "-itr"){
	assert(i+1 < argc);
	iteration = atoi(argv[i+1]);
	assert(minFreq > 0);
      }
      else if (arg == "-input"){
	assert(i+1 < argc);
	input = (std::string)argv[i+1];
      }
      else if (arg == "-output"){
	assert(i+1 < argc);
	output = (std::string)argv[i+1];
      }
    }
  }
};

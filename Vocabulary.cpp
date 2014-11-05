#include "Vocabulary.hpp"
#include "Utils.hpp"
#include <fstream>
#include <iostream>

Vocabulary::Vocabulary(const int wordVectorDim, const int contextLength, const int paragraphVectorDim):
  contextLen(contextLength),
  wordVecDim(wordVectorDim),
  paragraphVecDim(paragraphVectorDim),
  wordScoreVecDim(this->wordVecDim*this->contextLen+this->paragraphVecDim)
{}

void Vocabulary::read(const std::string& documentFile, COUNT freqThreshold){
  std::ifstream ifs(documentFile.c_str());
  std::unordered_map<std::string, INDEX> wordIndexTMP;
  std::vector<std::string> wordListTMP;
  std::vector<COUNT> wordCountTMP;
  COUNT unkCount = 0;
  INDEX paragraphIndex = 0;

  assert(ifs);

  for (std::string line, token; std::getline(ifs, line);){
    std::unordered_map<std::string, INDEX>::iterator it;
    bool tok = false;
    int beg = 0;

    ++paragraphIndex;

    for (int i = 0, len = line.length(); i < len; ++i){
      if (!tok && !Utils::isSpace(line[i])){
	beg = i;
	tok = true;
      }

      if (tok && (i == len-1 || Utils::isSpace(line[i]))){
	tok = false;
	token = (i == len-1) ? line.substr(beg, i-beg+1) : line.substr(beg, i-beg);
	it = wordIndexTMP.find(token);
	
	if (it != wordIndexTMP.end()){
	  wordCountTMP[it->second] += 1;
	}
	else {
	  wordIndexTMP[token] = wordListTMP.size();
	  wordListTMP.push_back(token);
	  wordCountTMP.push_back(1);
	}
      }
    }
  }

  COUNT totalCount = 0;

  for (int i = 0, size = wordListTMP.size(); i < size; ++i){
    if (wordCountTMP[i] < freqThreshold){
      unkCount += wordCountTMP[i];
      continue;
    }

    this->wordIndex[wordListTMP[i]] = this->wordList.size();
    this->wordList.push_back(wordListTMP[i]);
    this->wordCount.push_back(wordCountTMP[i]);
    this->discardProb.push_back((double)this->wordCount.back());
    totalCount += this->wordCount.back();

    for (COUNT j = 0, numNoise = (COUNT)pow(this->wordCount.back(), 0.75); j < numNoise; ++j){
      this->noiseDistribution.push_back(this->wordIndex.at(this->wordList.back()));
    }
  }

  totalCount += unkCount;

  for (int i = 0, size = this->discardProb.size(); i < size; ++i){
    this->discardProb[i] /= totalCount;
    this->discardProb[i] = 1.0-sqrt(1.0e-05/this->discardProb[i]);
  }

  std::unordered_map<std::string, INDEX>().swap(wordIndexTMP);
  std::vector<std::string>().swap(wordListTMP);
  std::vector<COUNT>().swap(wordCountTMP);
  
  this->unkIndex = this->wordList.size();
  this->nullIndex = this->unkIndex+1;
  this->wordList.push_back("**UNK**");
  this->wordList.push_back("**NULL**");
  this->wordVector = MatD::Random(this->wordVecDim, this->wordList.size()+2)*sqrt(6.0/(this->wordVecDim*2+1.0));
  this->paragraphVector = MatD::Random(this->paragraphVecDim, paragraphIndex)*sqrt(6.0/(this->paragraphVecDim*2+1.0));
  this->wordScoreVector = MatD::Zero(this->wordScoreVecDim, this->wordList.size());

  std::cout << "Documents: " << paragraphIndex << std::endl;
  std::cout << "Vocabulary size: " << this->wordList.size() << std::endl;
  std::cout << "Word embedding size: " << this->wordVecDim << std::endl;
  std::cout << "Paragraph embedding size: " << this->paragraphVecDim << std::endl;
  std::cout << "Context size: " << this->contextLen << std::endl;
}

void Vocabulary::train(const std::string& documentFile, const double learningRate, const int numNegative){
  std::ifstream ifs(documentFile.c_str());
  INDEX paragraphIndex = 0;

  for (std::string line, token; std::getline(ifs, line);){
    std::vector<INDEX> paragraph;
    std::unordered_map<std::string, INDEX>::iterator it;
    bool tok = false;
    int beg = 0;
    
    for (int i = 0; i < this->contextLen; ++i){
      paragraph.push_back(this->nullIndex);
    }
    
    for (int i = 0, len = line.length(); i < len; ++i){
      if (!tok && !Utils::isSpace(line[i])){
	beg = i;
	tok = true;
      }

      if (tok && (i == len-1 || Utils::isSpace(line[i]))){
	tok = false;
	token = (i == len-1) ? line.substr(beg, i-beg+1) : line.substr(beg, i-beg);
	it = this->wordIndex.find(token);
	
	if (it == this->wordIndex.end()){
	  paragraph.push_back(this->unkIndex);
	}
	else {
	  paragraph.push_back(it->second);
	}
      }
    }
    
    this->train(paragraphIndex++, paragraph, learningRate, numNegative);
    std::vector<INDEX>().swap(paragraph);
  }
}

void Vocabulary::train(const INDEX paragraphIndex, const std::vector<INDEX>& paragraph, const double learningRate, const int numNegative){
  MatD gradContext(this->wordVecDim, this->contextLen);
  MatD gradPara(this->paragraphVecDim, 1);
  std::unordered_map<INDEX, int> negHist;
  double deltaPos, deltaNeg;
  INDEX neg;

  for (int i = this->contextLen, size = paragraph.size(); i < size; ++i){
    if (paragraph[i] == this->unkIndex || this->discardProb[paragraph[i]] > Utils::uniformRandom()){
      continue;
    }
    
    gradContext.setZero();
    gradPara.setZero();
    negHist.clear();
    deltaPos = this->paragraphVector.col(paragraphIndex).dot(this->wordScoreVector.block(0, paragraph[i], this->paragraphVecDim, 1));

    for (int j = i-this->contextLen; j < i; ++j){
      deltaPos += this->wordVector.col(paragraph[j]).dot(this->wordScoreVector.block(this->paragraphVecDim+(j-i+this->contextLen)*this->wordVecDim, paragraph[i], this->wordVecDim, 1));
    }

    deltaPos = Utils::sigmoid(deltaPos)-1.0;
    gradPara = deltaPos*this->wordScoreVector.block(0, paragraph[i], this->paragraphVecDim, 1);

    for (int j = i-this->contextLen; j < i; ++j){
      gradContext.col(j-i+this->contextLen) = deltaPos*this->wordScoreVector.block(this->paragraphVecDim+(j-i+this->contextLen)*this->wordVecDim, paragraph[i], this->wordVecDim, 1);
    }

    deltaPos *= learningRate;
    this->wordScoreVector.block(0, paragraph[i], this->paragraphVecDim, 1) -= deltaPos*this->paragraphVector.col(paragraphIndex);

    for (int j = i-this->contextLen; j < i; ++j){
      this->wordScoreVector.block(this->paragraphVecDim+(j-i+this->contextLen)*this->wordVecDim, paragraph[i], this->wordVecDim, 1) -= deltaPos*this->wordVector.col(paragraph[j]);
    }

    for (int k = 0; k < numNegative; ++k){
      neg = paragraph[i];

      while (neg == paragraph[i] || negHist.count(neg)){
	neg = this->noiseDistribution[Utils::xor128()%this->noiseDistribution.size()];
      }

      negHist[neg] = 1;
      deltaNeg = this->paragraphVector.col(paragraphIndex).dot(this->wordScoreVector.block(0, neg, this->paragraphVecDim, 1));

      for (int j = i-this->contextLen; j < i; ++j){
	deltaNeg += this->wordVector.col(paragraph[j]).dot(this->wordScoreVector.block(this->paragraphVecDim+(j-i+this->contextLen)*this->wordVecDim, neg, this->wordVecDim, 1));
      }
      
      deltaNeg = Utils::sigmoid(deltaNeg);
      gradPara += deltaNeg*this->wordScoreVector.block(0, neg, this->paragraphVecDim, 1);
      
      for (int j = i-this->contextLen; j < i; ++j){
	gradContext.col(j-i+this->contextLen) += deltaNeg*this->wordScoreVector.block(this->paragraphVecDim+(j-i+this->contextLen)*this->wordVecDim, neg, this->wordVecDim, 1);
      }
      
      deltaNeg *= learningRate;
      this->wordScoreVector.block(0, neg, this->paragraphVecDim, 1) -= deltaNeg*this->paragraphVector.col(paragraphIndex);
      
      for (int j = i-this->contextLen; j < i; ++j){
	this->wordScoreVector.block(this->paragraphVecDim+(j-i+this->contextLen)*this->wordVecDim, neg, this->wordVecDim, 1) -= deltaNeg*this->wordVector.col(paragraph[j]);
      }
    }

    this->paragraphVector.col(paragraphIndex) -= learningRate*gradPara;

    for (int j = i-this->contextLen; j < i; ++j){
      this->wordVector.col(paragraph[j]) -= learningRate*gradContext.col(j-i+this->contextLen);
    }
  }

  std::unordered_map<INDEX, int>().swap(negHist);
}

void Vocabulary::outputParagraphVector(const std::string& fileName){
  std::ofstream ofs(fileName.c_str());

  for (int i = 0; i < this->paragraphVector.cols(); ++i){
    ofs << i;

    for (int j = 0; j < this->paragraphVector.rows(); ++j){
      ofs << " " << this->paragraphVector.coeff(j, i);
    }

    ofs << std::endl;
  }
}

void Vocabulary::outputWordVector(const std::string& fileName){
  std::ofstream ofs(fileName.c_str());

  for (int i = 0; i < this->wordVector.cols(); ++i){
    ofs << this->wordList[i];

    for (int j = 0; j < this->wordVector.rows(); ++j){
      ofs << " " << this->wordVector.coeff(j, i);
    }

    ofs << std::endl;
  }
}

void Vocabulary::save(const std::string& fileName){
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);
  double val = 0.0;

  assert(ofs);

  for (int i = 0; i < this->wordVector.cols(); ++i){
    for (int j = 0; j < this->wordVector.rows(); ++j){
      val = this->wordVector.coeff(j, i);
      Utils::infNan(val);
      ofs.write((char*)&val, sizeof(double));
    }
  }

  for (int i = 0; i < this->paragraphVector.cols(); ++i){
    for (int j = 0; j < this->paragraphVector.rows(); ++j){
      val = this->paragraphVector.coeff(j, i);
      Utils::infNan(val);
      ofs.write((char*)&val, sizeof(double));
    }
  }

  for (int i = 0; i < this->wordScoreVector.cols(); ++i){
    for (int j = 0; j < this->wordScoreVector.rows(); ++j){
      val = this->wordScoreVector.coeff(j, i);
      Utils::infNan(val);
      ofs.write((char*)&val, sizeof(double));
    }
  }
}

void Vocabulary::load(const std::string& fileName){
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);
  double val = 0.0;

  assert(ifs);

  for (int i = 0; i < this->wordVector.cols(); ++i){
    for (int j = 0; j < this->wordVector.rows(); ++j){
      ifs.read((char*)&val, sizeof(double));
      this->wordVector.coeffRef(j, i) = val;
      Utils::infNan(val);
    }
  }

  for (int i = 0; i < this->paragraphVector.cols(); ++i){
    for (int j = 0; j < this->paragraphVector.rows(); ++j){
      ifs.read((char*)&val, sizeof(double));
      this->paragraphVector.coeffRef(j, i) = val;
      Utils::infNan(val);
    }
  }

  for (int i = 0; i < this->wordScoreVector.cols(); ++i){
    for (int j = 0; j < this->wordScoreVector.rows(); ++j){
      ifs.read((char*)&val, sizeof(double));
      this->wordScoreVector.coeffRef(j, i) = val;
      Utils::infNan(val);
    }
  }
}

void Vocabulary::wordKnn(const int k){
  printf("KNN words of words\n");

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    if (!this->wordIndex.count(line)){
      continue;
    }

    INDEX target = this->wordIndex.at(line);
    MatD dist(1, this->wordList.size());

    for (INDEX i = 0; i < this->wordList.size(); ++i){
      dist.coeffRef(0, i) = (i == target ?
			     -1.0e+05:
			     Utils::cosDis(this->wordVector.col(target), this->wordVector.col(i)));
    }

    for (int i = 0; i < k; ++i){
      int row, col;

      dist.maxCoeff(&row, &col);
      dist.coeffRef(row, col) = -1.0e+05;
      printf("(%.5f) %s\n", Utils::cosDis(this->wordVector.col(col), this->wordVector.col(target)), this->wordList[col].c_str());
    }

    printf("\n");
  }
}

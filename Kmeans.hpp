#pragma once

#include "Matrix.hpp"

class Kmeans{
public:
  static void clustering(const int k, const MatD& sample, MatD& center, MatI& id, const int maxItr = 10);
};

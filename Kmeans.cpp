#include "Kmeans.hpp"

void Kmeans::clustering(const int k, const MatD& sample, MatD& center, MatI& id, const int maxItr){
  MatI count(1, k);

  center = MatD(sample.rows(), k);
  id = MatI(1, sample.cols());

  for (int i = 0; i < id.cols(); ++i){
    id.coeffRef(0, i) = i%k;
  }
  
  for (int itr = 0; itr < maxItr; ++itr){
    center.setZero();
    count.fill(1);

    for (int i = 0; i < sample.cols(); ++i){
      center.col(id.coeff(0, i)) += sample.col(i);
      count.coeffRef(0, id.coeff(0, i)) += 1;
    }

    for (int i = 0; i < center.cols(); ++i){
      center.col(i) /= count.coeff(0, i);
      center.col(i).normalize();
    }

    for (int i = 0; i < sample.cols(); ++i){
      double max = -1.0e+10, dot;
      int index = -1;

      for (int j = 0; j < center.cols(); ++j){
	dot = center.col(j).dot(sample.col(i));

	if (dot > max){
	  max = dot;
	  index = j;
	}
      }

      id.coeffRef(0, i) = index;
    }
  }

  center.setZero();
  count.fill(1);
  
  for (int i = 0; i < sample.cols(); ++i){
    center.col(id.coeff(0, i)) += sample.col(i);
    count.coeffRef(0, id.coeff(0, i)) += 1;
  }
  
  for (int i = 0; i < center.cols(); ++i){
    center.col(i) /= count.coeff(0, i);
    center.col(i).normalize();
  }
}

// Copyright 2014 by Kevin Duh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef _GMM_H_
#define _GMM_H_

#include <cassert>

/// Gaussian Mixture Model (GMM): prob(x) = \Prod_i weights[i] * Gaussian(x;means[i],covs[i])
class GMM {
 public:

  typedef std::unordered_map<unsigned, double> Word2Double;
  typedef std::unordered_map<unsigned, std::vector<double> > Word2Vector; 
  typedef std::vector<std::vector<double> > Matrix;

  GMM(){}
  GMM(unsigned numComponents, unsigned dim): 
  numComponents(numComponents), dim(dim), means(numComponents, std::vector<double>(dim)), 
    covs(numComponents, std::vector<double>(dim)), weights(numComponents,1/numComponents) {}

  void setComponent(const unsigned id, const std::vector<double>& mean, const std::vector<double>& cov){
    assert(id < numComponents);
    means[id] = mean;
    covs[id] = cov;
  }
  
  double prob(const std::vector<double>& sample) const {
    // return probability(sample) under this model (todo)
    double total = 0.0;
    for (unsigned id=0; id<numComponents; ++id){
      const std::vector<double>& m = means[id];
      const std::vector<double>& c = covs[id];
      double acc=0.0;
      double det=1.0;
      for (unsigned vi=0; vi<dim; ++vi){
	      acc += -0.5*std::pow(sample[vi]-m[vi],2)/c[vi];
	      det *= c[vi];
      }
      double normalizer = std::sqrt(std::pow(2*PI,dim)*std::fabs(det));
      double result = std::exp(acc)/normalizer;

      // temp: clipping to prevent nan; is there a better way?
      if (result < 1e-9)
	result = 1e-9;
      if (result > 1e+9)
	result = 1e+9;
      if (isnan(result))
	result = 1e+9; 

      total += weights[id]*result;
    }
    return total;
  }

  void MLE(const Word2Double& cpd, Word2Vector& embeddings){
    // temp: for now, assume 1 component
    double mass=0;
    std::vector<double> &m = means[0]; std::fill(m.begin(),m.end(),0.0);
    std::vector<double> &c = covs[0]; std::fill(c.begin(),c.end(),0.0);

    for (Word2Double::const_iterator j = cpd.begin(); j != cpd.end(); ++j) {
      std::vector<double> &e = embeddings[j->first];
      for (unsigned vi=0;vi<dim;++vi){ 
	m[vi] += j->second*e[vi]; 
	c[vi] += j->second*e[vi]*e[vi]; 
      }
      mass += j->second;
    }
    for (unsigned vi=0;vi<dim;++vi){ 
      m[vi] /= mass;  
      c[vi] = c[vi]/mass - m[vi]*m[vi]; 
    } 
  }
  
 private:
  unsigned numComponents; // number of mixture components
  unsigned dim; // dimension of each mean & diagonal covariance vector
  Matrix means; // matrix of means
  Matrix covs; // assume diagonal covariance
  std::vector<double> weights; // weights for each gaussian component
  static const double PI  = 3.141592653589793238463;

};


#endif

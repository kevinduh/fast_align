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
#include "kmeans.h"

/// Gaussian Mixture Model (GMM): prob(x) = \Prod_i weights[i] * Gaussian(x;means[i],covs[i])
class GMM {
 public:

  typedef std::unordered_map<unsigned, double> Word2Double;
  typedef std::unordered_map<unsigned, std::vector<double> > Word2Vector; 
  typedef std::vector<std::vector<double> > Matrix;

  GMM(){}
  GMM(unsigned k, unsigned dim): 
  numComponents(k), dim(dim), means(k, std::vector<double>(dim)), 
    covs(k, std::vector<double>(dim)), weights(k,1.0/(double)k) {}

  void setComponent(const unsigned id, const std::vector<double>& mean, const std::vector<double>& cov){
    assert(id < numComponents);
    means[id] = mean;
    covs[id] = cov;
  }
  
  void setWeights(const std::vector<double>& w){
    double sum = 0.0;
    for (unsigned i=0;i<w.size();++i){
      sum += w[i];
    }
    assert(sum>0.0);
    for (unsigned i=0;i<w.size();++i){
      weights[i] = w[i]/sum ;
    }
  }

  double prob(const std::vector<double>& sample) const {
    // return probability(sample) under this model 
    double total = 0.0;
    const std::vector<double> probs = probPerComponent(sample);
    for (unsigned id=0; id<numComponents; ++id){
      total += probs[id];
    }
    return total;
  }

  void MLE(const Word2Double& cpd, Word2Vector& embeddings){

    // E-step
    Word2Vector all_posterior; 
    for (Word2Double::const_iterator j = cpd.begin(); j != cpd.end(); ++j) {
      std::vector<double> &e = embeddings[j->first];
      std::vector<double> posterior = probPerComponent(e);
      double total = 0.0;
      for (unsigned id=0; id<numComponents; ++id)
	total += posterior[id];
      for (unsigned id=0; id<numComponents; ++id)
	posterior[id]/=total;
      all_posterior[j->first] = posterior;
    }

    // M-step
    double totalmass=0.0;
    for (unsigned id=0; id<numComponents; ++id){
      double mass=0;
      std::vector<double> &m = means[id]; std::fill(m.begin(),m.end(),0.0);
      std::vector<double> &c = covs[id]; std::fill(c.begin(),c.end(),0.00001);  // temp: add small regularizer to prevent instability

      for (Word2Double::const_iterator j = cpd.begin(); j != cpd.end(); ++j) {
	std::vector<double> &e = embeddings[j->first];
	for (unsigned vi=0;vi<dim;++vi){ 
	  m[vi] += j->second*all_posterior[j->first][id]*e[vi]; 
	  c[vi] += j->second*all_posterior[j->first][id]*e[vi]*e[vi]; 
	}
	mass += j->second*all_posterior[j->first][id];
      }
      if (mass > 0){
	for (unsigned vi=0;vi<dim;++vi){ 
	  m[vi] /= mass;  
	  c[vi] = c[vi]/mass - m[vi]*m[vi]; 
	} 
      }
      weights[id] = mass;
      totalmass += mass;
    }
    if (totalmass > 0){
      for (unsigned id=0; id<numComponents; ++id){
	weights[id] /= totalmass;
      }
    }
  }
  
  void print() const {
    std::cerr << "GMM weight: ";
    for (unsigned id=0; id < numComponents ; ++id)
      std::cerr << id << ":" << weights[id] << " ";
    for (unsigned id=0; id < numComponents ; ++id){
      std::cerr << std::endl << "  " << id << " mean: ";
      for (unsigned vi=0; vi < dim; ++vi)
	std::cerr << means[id][vi] << ", ";
      std::cerr << std::endl << "  " << id << " cov: ";
      for (unsigned vi=0; vi < dim; ++vi)
	std::cerr << covs[id][vi] << ", ";
    }
    std::cerr << std::endl;
  }

 private:

  std::vector<double> probPerComponent(const std::vector<double>& sample) const{
    std::vector<double> probs(numComponents,0.0);
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

      probs[id] = weights[id]*result;
    }
    return probs;
  }

  unsigned numComponents; // number of mixture components
  unsigned dim; // dimension of each mean & diagonal covariance vector
  Matrix means; // matrix of means
  Matrix covs; // assume diagonal covariance
  std::vector<double> weights; // weights for each gaussian component
  static const double PI  = 3.141592653589793238463;

};


#endif

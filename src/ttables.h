// Copyright 2013 by Chris Dyer
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
#ifndef _TTABLES_H_
#define _TTABLES_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "src/port.h"
#include "src/gmm.h"

struct Md {
  static double digamma(double x) {
    double result = 0, xx, xx2, xx4;
    for ( ; x < 7; ++x)
      result -= 1/x;
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
    return result;
  }
};

class TTable {
 public:
  TTable() {}
  typedef std::unordered_map<unsigned, double> Word2Double;
  typedef std::vector<Word2Double> Word2Word2Double;
  virtual ~TTable() {};

  virtual double prob(unsigned e, unsigned f) const = 0;
  virtual void NormalizeVB(const double alpha) = 0;
  virtual void Normalize() = 0;
  virtual TTable& operator+=(const TTable& rhs) = 0;
  virtual void ExportToFile(const char* filename, Dict& d) = 0;

  inline void Increment(unsigned e, unsigned f) {
    if (e >= counts.size()) counts.resize(e + 1);
    counts[e][f] += 1.0;
  }
  inline void Increment(unsigned e, unsigned f, double x) {
    if (e >= counts.size()) counts.resize(e + 1);
    counts[e][f] += x;
  }
 public:
  Word2Word2Double ttable;
  Word2Word2Double counts;
};


class MultinomialTable : public TTable {
 public:
  double prob(unsigned e, unsigned f) const {
    if (e < ttable.size()) {
      const Word2Double& cpd = ttable[e];
      const Word2Double::const_iterator it = cpd.find(f);
      if (it == cpd.end()) return 1e-9;
      return it->second;
    } else {
      return 1e-9;
    }
  }
  void NormalizeVB(const double alpha) {
    ttable.swap(counts);
    for (unsigned i = 0; i < ttable.size(); ++i) {
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        tot += it->second + alpha;
      if (!tot) tot = 1;
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second = exp(Md::digamma(it->second + alpha) - Md::digamma(tot));
    }
    counts.clear();
  }
  void Normalize() {
    ttable.swap(counts);
    for (unsigned i = 0; i < ttable.size(); ++i) {
      double tot = 0;
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        tot += it->second;
      if (!tot) tot = 1;
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it)
        it->second /= tot;
    }
    counts.clear();
  }
  // adds counts from another TTable - probabilities remain unchanged
  TTable& operator+=(const TTable& rhs) {
    if (rhs.counts.size() > counts.size()) counts.resize(rhs.counts.size());
    for (unsigned i = 0; i < rhs.counts.size(); ++i) {
      const Word2Double& cpd = rhs.counts[i];
      Word2Double& tgt = counts[i];
      for (Word2Double::const_iterator j = cpd.begin(); j != cpd.end(); ++j) {
        tgt[j->first] += j->second;
      }
    }
    return *this;
  }
  void ExportToFile(const char* filename, Dict& d) {
    std::ofstream file(filename);
    for (unsigned i = 0; i < ttable.size(); ++i) {
      const std::string& a = d.Convert(i);
      Word2Double& cpd = ttable[i];
      for (Word2Double::iterator it = cpd.begin(); it != cpd.end(); ++it) {
        const std::string& b = d.Convert(it->first);
        double c = log(it->second);
        file << a << '\t' << b << '\t' << c << std::endl;
      }
    }
    file.close();
  }

};


class GaussianTable : public TTable {
 public:

  typedef std::unordered_map<unsigned, std::vector<double> > Word2Vector; 

  GaussianTable(const std::string& embedding_file, Dict* d){
    // read file
    std::ifstream in(embedding_file.c_str());
    if (!in){
      std::cerr << "Can't read " << embedding_file << std::endl;
    }
    std::string line;
    in >> words >> dim; 
    std::cerr << "Reading embedding: "<< words << " words of dim " << dim << std::endl;
    std::vector<double> initial_mean(dim,0.0);
    std::vector<double> initial_covariance(dim,0.0);
    double num_embeddings = 0.0;
    while (getline(in,line)){
      std::string input_token;
      std::stringstream ss(line);
      getline(ss,input_token,' ');
      unsigned word_id = d->Convert(input_token,true);
      if (word_id > 0){
	++num_embeddings;
	embeddings[word_id] = std::vector<double>();
	for (unsigned vi=0;vi<dim;++vi){
	  getline(ss,input_token,' ');
	  double value = ::atof(input_token.c_str());
	  embeddings[word_id].push_back(value);
	  initial_mean[vi] += value;
	  initial_covariance[vi] += (value*value);
	}
      }
    }

    // initialize gaussian mean and covariance
    for (unsigned vi=0;vi<dim;++vi){
      initial_mean[vi] /= num_embeddings;
      initial_covariance[vi] = initial_covariance[vi]/num_embeddings - initial_mean[vi]*initial_mean[vi];
    }
    for (unsigned v=0; v <= d->max(); ++v){
      // assign same GMM to both e/f sides, even though in practice we only need to index on e
      // further, initialize with 1-component GMM
      gmms[v] = GMM(1,dim);
      gmms[v].setComponent(0,initial_mean,initial_covariance);
    }

    d_ = d; //temp
    //PrintEmbeddings(d); //temp

    // Assign default vector to all words that do not have embeddings
    for (unsigned v=0; v <= d->max(); ++v){
      Word2Vector::const_iterator it = embeddings.find(v);
      if (it == embeddings.end()){
	embeddings[v] = initial_mean; // temp: is mean vector the best?
      }
    }
  }

  double prob(unsigned e, unsigned f) const {
    const std::vector<double> & fvec = embeddings.find(f)->second;
    GMM gmm = gmms.find(e)->second;
    double result = gmm.prob(fvec);
    return result;
  }

  void NormalizeVB(const double alpha) {
    std::cout << "Not yet implemented";
    exit(1);
  }

  void Normalize() {
    /* for (unsigned i = 0; i < counts.size(); ++i) { */
    /*   const Word2Double& cpd = counts[i]; */
    /*   double mass=0; */
    /*   std::vector<double> &m = mean[i]; */
    /*   std::fill(m.begin(),m.end(),0.0); */
    /*   std::vector<double> &c = covariance[i]; */
    /*   std::fill(c.begin(),c.end(),0.0); */
    /*   for (Word2Double::const_iterator j = cpd.begin(); j != cpd.end(); ++j) { */
    /* 	std::vector<double> &e = embeddings[j->first]; */
    /* 	for (unsigned vi=0;vi<dim;++vi){ */
    /* 	  m[vi] += j->second*e[vi]; */
    /* 	  c[vi] += j->second*e[vi]*e[vi]; */
    /* 	} */
    /* 	mass += j->second; */
    /*   } */

    /*   for (unsigned vi=0;vi<dim;++vi){ */
    /* 	m[vi] /= mass;  */
    /* 	c[vi] = c[vi]/mass - m[vi]*m[vi]; */
    /*   } */
    /*   if (mass != 0 && i%100==0) {PrintMean(i);} //temp */
    /* } */

    /// TEMP: FOR NOW, no updating
    for (unsigned i = 0; i < counts.size(); ++i) { 
      gmms[i].MLE(counts[i],embeddings);
    }


    counts.clear();
  }

  TTable& operator+=(const TTable& rhs) {
    std::cout << "Not yet implemented";
    exit(1);

  }

  void ExportToFile(const char* filename, Dict& d) {
    std::cout << "Not yet implemented";
    exit(1);
  }

  void PrintVector(const std::vector<double> &v, const std::string tag) const{
    std::cerr << tag << " vec: [";
    for (unsigned vi=0;vi<dim;++vi){
      std::cerr << v[vi] << ", ";
    }
    std::cerr << "]" << std::endl;
  }
  void PrintEmbeddings(Dict* d){
    for (unsigned v=0; v <= d->max(); ++v){
      std::string this_word = d->Convert(v);
      Word2Vector::const_iterator it = embeddings.find(v);
      if (it == embeddings.end()){
	std::cerr << "word/id: " << this_word << " " << v << " __no_embedding__ ";
      }
      else {
	std::cerr << "word/id: " << this_word << " " << v << " " ;
	for (unsigned vv=0;vv<dim;++vv){
	  std::cerr << (it->second)[vv] << " ";
	}      
      }
      std::cerr << std::endl;
    }

  }

 private:
  Word2Vector embeddings; // collection of word embeddings
  std::unordered_map< unsigned, GMM > gmms; // collection of Gaussian mixture models
  unsigned words; // number of words in embeddings file
  unsigned dim; // dimension of embedding vectors
  Dict* d_; //temp
};

#endif

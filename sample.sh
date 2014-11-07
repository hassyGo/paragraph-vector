wget http://www.logos.t.u-tokyo.ac.jp/~hassy/implementations/paragraph_vector/wikipedia.sample.doc
wget http://www.logos.t.u-tokyo.ac.jp/~hassy/implementations/paragraph_vector/wikipedia.sample.title
make
time ./paragraph_vector -input ./wikipedia.sample.doc -output sample -wvdim 50 -pvdim 100 -neg 5 -itr 10

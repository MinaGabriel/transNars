#!/bin/bash

# Step 1: Clone the repository
git clone https://github.com/thunlp/OpenKE.git
 
cd OpenKE
 
mkdir ../benchmarks 
mv benchmarks/* ../benchmarks
 
rmdir ./benchmarks
cd ..
rm -rf ./OpenKE
echo "Repository cleaned and benchmarks folder moved up."
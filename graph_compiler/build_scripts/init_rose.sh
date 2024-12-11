# source before building 
export PREFIX=     CHOOSE_ROSE_INSTALLATION_LOCATION
export ROSE_INS=$PREFIX/install
export PATH=$ROSE_INS/bin:$PATH
export LD_LIBRARY_PATH=$ROSE_INS/lib:$LD_LIBRARY_PATH
export BOOST_ROOT=/usr/local
export LD_LIBRARY_PATH=$BOOST_ROOT/lib:$LD_LIBRARY_PATH

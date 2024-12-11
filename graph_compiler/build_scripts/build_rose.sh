export PREFIX=   CHOOSE_ROSE_INSTALLATION_LOCATION
export NUM_PROCESSORS=12

mkdir rose

apt update
apt install build-essential wget make xz-utils git automake libtool flex bison -y

cd $PREFIX

wget -nv https://archives.boost.io/release/1.67.0/source/boost_1_67_0.tar.gz
tar -xzf boost_1_67_0.tar.gz
rm boost_1_67_0.tar.gz
cd boost_1_67_0
./bootstrap.sh --with-libraries=atomic,chrono,date_time,filesystem,iostreams,program_options,random,regex,serialization,signals,system,thread,wave
./b2 -std=c++11 install

export LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH}"
export BOOST_ROOT="/usr/local/"

git clone https://github.com/rose-compiler/rose "${PREFIX}/src"
cd "${PREFIX}/src"
git checkout edeaed4
./build
mkdir ../build
cd ../build
../src/configure --prefix="${PREFIX}/install" \
                 --enable-languages=c,c++ \
                 --with-boost="${BOOST_ROOT}"
make core -j${NUM_PROCESSORS}
make install-core -j${NUM_PROCESSORS}
make check-core -j${NUM_PROCESSORS}

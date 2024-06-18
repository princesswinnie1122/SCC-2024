OPTFLAGS="-march=cascadelake -O2"
CC=mpicc
CXX=mpicxx
FC=mpifort

CFLAGS="-fPIC -march=cascadelake -fopenmp -Wrestrict -DLMP_INTEL_USELRT -DLMP_USE_MKL_RNG -Df2cFortran $OPTFLAGS"
CXXFLAGS="-fPIC -march=cascadelake -fopenmp -Wrestrict -DLMP_INTEL_USELRT -DLMP_USE_MKL_RNG -Df2cFortran -std=c++17 $OPTFLAGS"
FCFLAGS="-fPIC -march=cascadelake -Wrestrict -DLMP_INTEL_USELRT -DLMP_USE_MKL_RNG $OPTFLAGS"
LDFLAGS="$OPTFLAGS -L$MKLROOT/lib/intel64 -ltbbmalloc -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -L$ZFP_DIR/lib -lzfp"

cmake ../cmake \
-DCMAKE_INSTALL_PREFIX=$BUILD/$LAMMPS \
-DBUILD_SHARED_LIBS=yes \
-DINTEL_ARCH=cpu \
-DFFT=MKL \
-DFFT_MKL_THREADS=on \
-DWITH_GZIP=yes \
-DPKG_ADIOS=yes \
-DADIOS2_DIR=$ADIOS_DIR \
-DPKG_NETCDF=yes \
-DBUILD_MPI=yes \
-DBUILD_OMP=yes \
-DLAMMPS_MACHINE=mpi \
-DPKG_OPENMP=yes \
-DPKG_OPT=yes \
-DPYTHON_EXECUTABLE=$(which python3) \
-DPKG_ML-QUIP=yes \
-DPKG_ML-HDNNP=yes \
-DPKG_VORONOI=yes \
-DWITH_JPEG=yes \
-DLAMMPS_FFMPEG=yes

make -j
make install
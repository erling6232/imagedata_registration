#
# muod=multigrid_nonlin_cy

sitepackages := $(shell python -c 'import site; print(site.getsitepackages()[0])')
pythonincludes := $(shell python -c 'import sysconfig; info=sysconfig.get_paths(); print(info["include"])')

all:	multigrid_nonlin_cy.so multigrid_nonlin_cc.so

#    -fopenmp
#	-msse3 -O3 \
#	-fopenmp -ffast-math -lm -mavx2 \

multigrid_nonlin_cy.so:	multigrid_nonlin_cy.c
	gcc -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing \
	-O2 \
	-ffast-math \
	-march=native \
	-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
	-I$(sitepackages)/numpy/core/include \
	-I$(sitepackages)/numpy/_core/include \
	-I$(sitepackages)/numpy \
	-I$(pythonincludes) \
	-ftree-loop-vectorize -ffast-math -lm -mavx \
	-o $@ $<

multigrid_nonlin_cy.c:	multigrid_nonlin_cy.pyx
	python3 -m cython \
	-I$(sitepackages)/numpy \
	-a \
	$<

.PHONY = clean
clean:
	rm -f multigrid_nonlin_cy.c multigrid_nonlin_cy.so
	rm -f multigrid_nonlin_cc.c multigrid_nonlin_cc.so

#pythran -DUSE_BOOST_SIMD -march=native  navlam_nonlinear_highlevel.py

multigrid_nonlin_cc.so:	multigrid_nonlin_cc.c navlam_nonlinear_cc.o
	gcc -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing \
	-O3 -fno-trapping-math -fno-math-errno -fno-signed-zeros -ftree-loop-vectorize \
	-mavx512f \
	-march=native \
	-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
	-I$(sitepackages)/numpy \
	-I$(pythonincludes) \
	-lm \
	navlam_nonlinear_cc.o \
	-o $@ $<

multigrid_nonlin_cc.c:	multigrid_nonlin_cc.pyx
	python3 -m cython \
	-I$(sitepackages)/numpy \
	-a \
	$<

navlam_nonlinear_cc.o:	navlam_nonlinear_cc.c
	gcc -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing \
	-O3 -mavx512f \
	-fno-trapping-math -fno-math-errno -fno-signed-zeros -ftree-loop-vectorize \
	-march=native \
	-c $<


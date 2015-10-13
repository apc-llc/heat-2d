ARCH=35

#include ../common.mk

all: heat-2d

heat-2d_host.cu heat-2d_kernel.cu: heat-2d.c
	ppcg $<

heat-2d: heat-2d_host.cu heat-2d_kernel.cu
	nvcc -DTIME -DVERIFY -arch=sm_$(ARCH) $^ -o $@

clean:
	rm -rf heat-2d_host.cu heat-2d_kernel.cu heat-2d


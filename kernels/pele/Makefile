###############################################################################
#Copyright (c) 2023 Advanced Micro Devices, Inc. (AMD)
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
###############################################################################

AMD_ARCH=gfx942
ROCM_PATH ?=/opt/rocm
CC = ${ROCM_PATH}/bin/hipcc
CFLAGS = -MMD -MP -std=c++17 -m64 --offload-arch=${AMD_ARCH} -pthread -g -O3 -munsafe-fp-atomics -I${ROCM_PATH}/include/

#XAMPLES =  pelec_repro2_LiDryer pelec_repro2_drm19 pelec_repro2_dodecane_lu pelec_repro2_isooctane_lu
EXAMPLES = pelec_repro2_dodecane_lu

all: $(EXAMPLES)

%.o: %.cpp
	$(CC) $(CFLAGS) $(FLAGS) -o $@ -c $<

pelec_repro2_LiDryer: pelec_repro2_LiDryer.o
	$(CC) -o $@ $@.o

pelec_repro2_drm19: pelec_repro2_drm19.o
	$(CC) -o $@ $@.o

pelec_repro2_dodecane_lu: pelec_repro2_dodecane_lu.o
	$(CC) -o $@ $@.o

pelec_repro2_isooctane_lu: pelec_repro2_isooctane_lu.o
	$(CC) -o $@ $@.o

clean:
	rm -f ${EXAMPLES} *.o *~ *unknown* *amdgcn* *.d*

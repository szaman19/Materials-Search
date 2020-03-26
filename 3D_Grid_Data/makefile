NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))

ifeq ($(NVCC_TEST),nvcc)
		CC := nvcc
		VEXT := Vext_gpu.cu	
else
		CC := gcc
		VEXT := Vext_cpu.c	

endif

VEXT_OBJ := $(addsuffix .o,$(basename $(VEXT)))

inp_compile: cif2input.c
	gcc -c cif2input.c 

inp_link: inp_compile
	gcc cif2input.o -o cif2input

inp_test: inp_link
	./cif2input example.cif data_ff_UFF example.inp

vext_compile: $(VEXT)
	$(CC) -c $(VEXT)

vext_link: vext_compile
	$(CC) $(VEXT_OBJ) -lm -o ./egrid

vext_test: vext_link
	./egrid example.inp example.grid


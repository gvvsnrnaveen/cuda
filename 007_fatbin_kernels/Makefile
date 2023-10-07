INCLUDES=-I../Common
LIBS+=-lcuda -lcudart_static

all:
	nvcc vector_addition.cu -o vector_addition.fatbin -fatbin
	nvcc vector_multiplication.cu -o vector_multiplication.fatbin -fatbin
	nvcc vector_arithmetic.cu -o vector_arithmetic.fatbin -fatbin
	
	nvcc -ccbin g++ main.cpp -o fatbin_kernel -m64 -std=c++11 $(INCLUDES) $(LIBS)

clean:
	rm -rf vector_addition.fatbin vector_multiplication.fatbin vector_arithmetic.fatbin fatbin_kernel 

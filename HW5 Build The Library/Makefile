CC = g++
CFLAGS = -Wall -fPIC
exe = main.elf
obj = Merge.o Mergesort.o PrintArray.o

all: $(exe)

# Shared library
libmerge.so: $(obj)
	$(CC) -shared -o libmerge.so $(obj)

# Static library
libmerge.a: $(obj)
	ar rcs libmerge.a $(obj) 

# Object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Executable
$(exe): main.cpp libmerge.so libmerge.a
	$(CC) main.cpp -L. -lmerge -o $(exe)

clean:
	rm -f $(obj) libmerge.so libmerge.a $(exe)

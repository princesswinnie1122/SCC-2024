NVCXX = nvc++
NVCXXFLAGS = -O0 -std=c++14 -cuda -gpu=cc60 -Minfo=all
MNIST=-I/home/pp23/share/lab5/testcases/weights/mnist/include
TARGETS = lab5

.PHONY: all
all: $(TARGETS)

%: %.cpp
	$(NVCXX) $(NVCXXFLAGS) $(MNIST) -o $@ $<

clean:
	rm -f $(TARGETS)

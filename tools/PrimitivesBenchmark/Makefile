HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=PrimitivesBenchmark
CXXFLAGS = -O3 -g -I../../src/include -I.

all: $(EXE)

$(EXE): PrimitivesBenchmark.cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o $(EXE)

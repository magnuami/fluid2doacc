# PGI compiler flags used:
# -fast						A generally optimal set of options including global optimization, 
#							SIMD vectorization, look unrolling and cache optimizations. 
# -Mipa=fast,inline			Agressive inte-procedural analysis and optimization, including automatic inlining.
# -acc 						Compile using OpenACC
# -ta=nvidia,keepgpu,time	Set the target to nvidia accelerators, keep intermediate cuda code, 
#							and use a simple timing mechanism on the code.
# -Minfo 					Give detailed information about the compilation

CC       = pgcc
CCFLAGS  = -fast -Mipa=fast
LDFLAGS  = -lGLEW -lGL -lXext -lX11 -lGLU -lglut -lglfw
ACCFLAGS = -acc -ta=nvidia -Minfo
#ACCFLAGS = -acc -ta=nvidia,keepgpu,time -Minfo
#ACCFLAGS = -acc -ta=nvidia

CSRC = fluid2d.c main.c 
OBJECTS = $(CSRC:.c=.o)
BIN = fluid2dacc

all: $(BIN)

run: $(BIN)
	./fluid2dacc

$(BIN): $(CSRC)
	$(CC) $(CCFLAGS) $(ACCFLAGS) $(LDFLAGS) -o $@ $(CSRC)

clean:
	$(RM) $(BIN) *.o
















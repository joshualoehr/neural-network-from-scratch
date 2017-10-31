CXX=g++
RM=rm -f
CPPFLAGS=-g -pthread -Wall -I ./eigen/ 
LDFLAGS=-g

SRCS=ann.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: ann

ann: ann.cpp
	$(CXX) $(CPPFLAGS) -o ann ann.cpp

# ann.o: ann.cpp
#	$(CXX) $(CPPFLAGS) ann.cpp -o ann.o

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) ann

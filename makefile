# Adapted from: https://spin.atomicobject.com/2016/08/26/makefile-c-projects/

CXX=g++
CPPFLAGS=-g -Wall -I ./eigen/

TARGET ?= ann
SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I, $(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LOADLIBES) $(LDLIBS)

.PHONY: clean

clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS)


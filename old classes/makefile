CC 				:= g++
CXX				:= c++
CXXFLAGS	+= -Wall -O -std=c++0x

SRCDIR		:= src
BUILDDIR	:= build
TARGET 		:= bin/runner

SRCEXT 		:= cpp
SOURCES		:= $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS 	:= $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
LDLIBS		+= -larmadillo -lpthread -lstdc++ -lm
#LDLIBS		+= -L/home/frederik/gsl/lib -lgsl -lgslcblas
INC				:= -I include #/home/frederik/gsl/include

$(TARGET): $(OBJECTS)
	@echo " $(CC) $^ -o $(TARGET) $(LDLIBS)"; $(CC) $^ -o $(TARGET) $(LDLIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CXXFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CXXFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)

.PHONY: clean

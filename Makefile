# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/frederik/SVMprojekt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frederik/SVMprojekt

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/frederik/SVMprojekt/CMakeFiles /home/frederik/SVMprojekt/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/frederik/SVMprojekt/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named HeliumAtom

# Build rule for target.
HeliumAtom: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 HeliumAtom
.PHONY : HeliumAtom

# fast build rule for target.
HeliumAtom/fast:
	$(MAKE) -f CMakeFiles/HeliumAtom.dir/build.make CMakeFiles/HeliumAtom.dir/build
.PHONY : HeliumAtom/fast

#=============================================================================
# Target rules for targets named TwoParticleSqueeze

# Build rule for target.
TwoParticleSqueeze: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 TwoParticleSqueeze
.PHONY : TwoParticleSqueeze

# fast build rule for target.
TwoParticleSqueeze/fast:
	$(MAKE) -f CMakeFiles/TwoParticleSqueeze.dir/build.make CMakeFiles/TwoParticleSqueeze.dir/build
.PHONY : TwoParticleSqueeze/fast

#=============================================================================
# Target rules for targets named SVMlib

# Build rule for target.
SVMlib: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 SVMlib
.PHONY : SVMlib

# fast build rule for target.
SVMlib/fast:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/build
.PHONY : SVMlib/fast

mainfiles/HeliumAtom.o: mainfiles/HeliumAtom.cpp.o

.PHONY : mainfiles/HeliumAtom.o

# target to build an object file
mainfiles/HeliumAtom.cpp.o:
	$(MAKE) -f CMakeFiles/HeliumAtom.dir/build.make CMakeFiles/HeliumAtom.dir/mainfiles/HeliumAtom.cpp.o
.PHONY : mainfiles/HeliumAtom.cpp.o

mainfiles/HeliumAtom.i: mainfiles/HeliumAtom.cpp.i

.PHONY : mainfiles/HeliumAtom.i

# target to preprocess a source file
mainfiles/HeliumAtom.cpp.i:
	$(MAKE) -f CMakeFiles/HeliumAtom.dir/build.make CMakeFiles/HeliumAtom.dir/mainfiles/HeliumAtom.cpp.i
.PHONY : mainfiles/HeliumAtom.cpp.i

mainfiles/HeliumAtom.s: mainfiles/HeliumAtom.cpp.s

.PHONY : mainfiles/HeliumAtom.s

# target to generate assembly for a file
mainfiles/HeliumAtom.cpp.s:
	$(MAKE) -f CMakeFiles/HeliumAtom.dir/build.make CMakeFiles/HeliumAtom.dir/mainfiles/HeliumAtom.cpp.s
.PHONY : mainfiles/HeliumAtom.cpp.s

mainfiles/TwoParticleSqueeze.o: mainfiles/TwoParticleSqueeze.cpp.o

.PHONY : mainfiles/TwoParticleSqueeze.o

# target to build an object file
mainfiles/TwoParticleSqueeze.cpp.o:
	$(MAKE) -f CMakeFiles/TwoParticleSqueeze.dir/build.make CMakeFiles/TwoParticleSqueeze.dir/mainfiles/TwoParticleSqueeze.cpp.o
.PHONY : mainfiles/TwoParticleSqueeze.cpp.o

mainfiles/TwoParticleSqueeze.i: mainfiles/TwoParticleSqueeze.cpp.i

.PHONY : mainfiles/TwoParticleSqueeze.i

# target to preprocess a source file
mainfiles/TwoParticleSqueeze.cpp.i:
	$(MAKE) -f CMakeFiles/TwoParticleSqueeze.dir/build.make CMakeFiles/TwoParticleSqueeze.dir/mainfiles/TwoParticleSqueeze.cpp.i
.PHONY : mainfiles/TwoParticleSqueeze.cpp.i

mainfiles/TwoParticleSqueeze.s: mainfiles/TwoParticleSqueeze.cpp.s

.PHONY : mainfiles/TwoParticleSqueeze.s

# target to generate assembly for a file
mainfiles/TwoParticleSqueeze.cpp.s:
	$(MAKE) -f CMakeFiles/TwoParticleSqueeze.dir/build.make CMakeFiles/TwoParticleSqueeze.dir/mainfiles/TwoParticleSqueeze.cpp.s
.PHONY : mainfiles/TwoParticleSqueeze.cpp.s

src/CMAES.o: src/CMAES.cpp.o

.PHONY : src/CMAES.o

# target to build an object file
src/CMAES.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/CMAES.cpp.o
.PHONY : src/CMAES.cpp.o

src/CMAES.i: src/CMAES.cpp.i

.PHONY : src/CMAES.i

# target to preprocess a source file
src/CMAES.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/CMAES.cpp.i
.PHONY : src/CMAES.cpp.i

src/CMAES.s: src/CMAES.cpp.s

.PHONY : src/CMAES.s

# target to generate assembly for a file
src/CMAES.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/CMAES.cpp.s
.PHONY : src/CMAES.cpp.s

src/CoulombPotential.o: src/CoulombPotential.cpp.o

.PHONY : src/CoulombPotential.o

# target to build an object file
src/CoulombPotential.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/CoulombPotential.cpp.o
.PHONY : src/CoulombPotential.cpp.o

src/CoulombPotential.i: src/CoulombPotential.cpp.i

.PHONY : src/CoulombPotential.i

# target to preprocess a source file
src/CoulombPotential.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/CoulombPotential.cpp.i
.PHONY : src/CoulombPotential.cpp.i

src/CoulombPotential.s: src/CoulombPotential.cpp.s

.PHONY : src/CoulombPotential.s

# target to generate assembly for a file
src/CoulombPotential.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/CoulombPotential.cpp.s
.PHONY : src/CoulombPotential.cpp.s

src/DoubleTrapPotential.o: src/DoubleTrapPotential.cpp.o

.PHONY : src/DoubleTrapPotential.o

# target to build an object file
src/DoubleTrapPotential.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/DoubleTrapPotential.cpp.o
.PHONY : src/DoubleTrapPotential.cpp.o

src/DoubleTrapPotential.i: src/DoubleTrapPotential.cpp.i

.PHONY : src/DoubleTrapPotential.i

# target to preprocess a source file
src/DoubleTrapPotential.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/DoubleTrapPotential.cpp.i
.PHONY : src/DoubleTrapPotential.cpp.i

src/DoubleTrapPotential.s: src/DoubleTrapPotential.cpp.s

.PHONY : src/DoubleTrapPotential.s

# target to generate assembly for a file
src/DoubleTrapPotential.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/DoubleTrapPotential.cpp.s
.PHONY : src/DoubleTrapPotential.cpp.s

src/MatrixElements.o: src/MatrixElements.cpp.o

.PHONY : src/MatrixElements.o

# target to build an object file
src/MatrixElements.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/MatrixElements.cpp.o
.PHONY : src/MatrixElements.cpp.o

src/MatrixElements.i: src/MatrixElements.cpp.i

.PHONY : src/MatrixElements.i

# target to preprocess a source file
src/MatrixElements.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/MatrixElements.cpp.i
.PHONY : src/MatrixElements.cpp.i

src/MatrixElements.s: src/MatrixElements.cpp.s

.PHONY : src/MatrixElements.s

# target to generate assembly for a file
src/MatrixElements.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/MatrixElements.cpp.s
.PHONY : src/MatrixElements.cpp.s

src/Multidim_min.o: src/Multidim_min.cpp.o

.PHONY : src/Multidim_min.o

# target to build an object file
src/Multidim_min.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Multidim_min.cpp.o
.PHONY : src/Multidim_min.cpp.o

src/Multidim_min.i: src/Multidim_min.cpp.i

.PHONY : src/Multidim_min.i

# target to preprocess a source file
src/Multidim_min.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Multidim_min.cpp.i
.PHONY : src/Multidim_min.cpp.i

src/Multidim_min.s: src/Multidim_min.cpp.s

.PHONY : src/Multidim_min.s

# target to generate assembly for a file
src/Multidim_min.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Multidim_min.cpp.s
.PHONY : src/Multidim_min.cpp.s

src/PotentialList.o: src/PotentialList.cpp.o

.PHONY : src/PotentialList.o

# target to build an object file
src/PotentialList.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/PotentialList.cpp.o
.PHONY : src/PotentialList.cpp.o

src/PotentialList.i: src/PotentialList.cpp.i

.PHONY : src/PotentialList.i

# target to preprocess a source file
src/PotentialList.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/PotentialList.cpp.i
.PHONY : src/PotentialList.cpp.i

src/PotentialList.s: src/PotentialList.cpp.s

.PHONY : src/PotentialList.s

# target to generate assembly for a file
src/PotentialList.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/PotentialList.cpp.s
.PHONY : src/PotentialList.cpp.s

src/PotentialStrategy.o: src/PotentialStrategy.cpp.o

.PHONY : src/PotentialStrategy.o

# target to build an object file
src/PotentialStrategy.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/PotentialStrategy.cpp.o
.PHONY : src/PotentialStrategy.cpp.o

src/PotentialStrategy.i: src/PotentialStrategy.cpp.i

.PHONY : src/PotentialStrategy.i

# target to preprocess a source file
src/PotentialStrategy.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/PotentialStrategy.cpp.i
.PHONY : src/PotentialStrategy.cpp.i

src/PotentialStrategy.s: src/PotentialStrategy.cpp.s

.PHONY : src/PotentialStrategy.s

# target to generate assembly for a file
src/PotentialStrategy.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/PotentialStrategy.cpp.s
.PHONY : src/PotentialStrategy.cpp.s

src/SingleGaussPotential.o: src/SingleGaussPotential.cpp.o

.PHONY : src/SingleGaussPotential.o

# target to build an object file
src/SingleGaussPotential.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/SingleGaussPotential.cpp.o
.PHONY : src/SingleGaussPotential.cpp.o

src/SingleGaussPotential.i: src/SingleGaussPotential.cpp.i

.PHONY : src/SingleGaussPotential.i

# target to preprocess a source file
src/SingleGaussPotential.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/SingleGaussPotential.cpp.i
.PHONY : src/SingleGaussPotential.cpp.i

src/SingleGaussPotential.s: src/SingleGaussPotential.cpp.s

.PHONY : src/SingleGaussPotential.s

# target to generate assembly for a file
src/SingleGaussPotential.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/SingleGaussPotential.cpp.s
.PHONY : src/SingleGaussPotential.cpp.s

src/System.o: src/System.cpp.o

.PHONY : src/System.o

# target to build an object file
src/System.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/System.cpp.o
.PHONY : src/System.cpp.o

src/System.i: src/System.cpp.i

.PHONY : src/System.i

# target to preprocess a source file
src/System.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/System.cpp.i
.PHONY : src/System.cpp.i

src/System.s: src/System.cpp.s

.PHONY : src/System.s

# target to generate assembly for a file
src/System.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/System.cpp.s
.PHONY : src/System.cpp.s

src/TrapPotential.o: src/TrapPotential.cpp.o

.PHONY : src/TrapPotential.o

# target to build an object file
src/TrapPotential.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/TrapPotential.cpp.o
.PHONY : src/TrapPotential.cpp.o

src/TrapPotential.i: src/TrapPotential.cpp.i

.PHONY : src/TrapPotential.i

# target to preprocess a source file
src/TrapPotential.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/TrapPotential.cpp.i
.PHONY : src/TrapPotential.cpp.i

src/TrapPotential.s: src/TrapPotential.cpp.s

.PHONY : src/TrapPotential.s

# target to generate assembly for a file
src/TrapPotential.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/TrapPotential.cpp.s
.PHONY : src/TrapPotential.cpp.s

src/Utils.o: src/Utils.cpp.o

.PHONY : src/Utils.o

# target to build an object file
src/Utils.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Utils.cpp.o
.PHONY : src/Utils.cpp.o

src/Utils.i: src/Utils.cpp.i

.PHONY : src/Utils.i

# target to preprocess a source file
src/Utils.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Utils.cpp.i
.PHONY : src/Utils.cpp.i

src/Utils.s: src/Utils.cpp.s

.PHONY : src/Utils.s

# target to generate assembly for a file
src/Utils.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Utils.cpp.s
.PHONY : src/Utils.cpp.s

src/Variational.o: src/Variational.cpp.o

.PHONY : src/Variational.o

# target to build an object file
src/Variational.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Variational.cpp.o
.PHONY : src/Variational.cpp.o

src/Variational.i: src/Variational.cpp.i

.PHONY : src/Variational.i

# target to preprocess a source file
src/Variational.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Variational.cpp.i
.PHONY : src/Variational.cpp.i

src/Variational.s: src/Variational.cpp.s

.PHONY : src/Variational.s

# target to generate assembly for a file
src/Variational.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/Variational.cpp.s
.PHONY : src/Variational.cpp.s

src/fdcube.o: src/fdcube.cpp.o

.PHONY : src/fdcube.o

# target to build an object file
src/fdcube.cpp.o:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/fdcube.cpp.o
.PHONY : src/fdcube.cpp.o

src/fdcube.i: src/fdcube.cpp.i

.PHONY : src/fdcube.i

# target to preprocess a source file
src/fdcube.cpp.i:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/fdcube.cpp.i
.PHONY : src/fdcube.cpp.i

src/fdcube.s: src/fdcube.cpp.s

.PHONY : src/fdcube.s

# target to generate assembly for a file
src/fdcube.cpp.s:
	$(MAKE) -f CMakeFiles/SVMlib.dir/build.make CMakeFiles/SVMlib.dir/src/fdcube.cpp.s
.PHONY : src/fdcube.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... HeliumAtom"
	@echo "... TwoParticleSqueeze"
	@echo "... SVMlib"
	@echo "... rebuild_cache"
	@echo "... mainfiles/HeliumAtom.o"
	@echo "... mainfiles/HeliumAtom.i"
	@echo "... mainfiles/HeliumAtom.s"
	@echo "... mainfiles/TwoParticleSqueeze.o"
	@echo "... mainfiles/TwoParticleSqueeze.i"
	@echo "... mainfiles/TwoParticleSqueeze.s"
	@echo "... src/CMAES.o"
	@echo "... src/CMAES.i"
	@echo "... src/CMAES.s"
	@echo "... src/CoulombPotential.o"
	@echo "... src/CoulombPotential.i"
	@echo "... src/CoulombPotential.s"
	@echo "... src/DoubleTrapPotential.o"
	@echo "... src/DoubleTrapPotential.i"
	@echo "... src/DoubleTrapPotential.s"
	@echo "... src/MatrixElements.o"
	@echo "... src/MatrixElements.i"
	@echo "... src/MatrixElements.s"
	@echo "... src/Multidim_min.o"
	@echo "... src/Multidim_min.i"
	@echo "... src/Multidim_min.s"
	@echo "... src/PotentialList.o"
	@echo "... src/PotentialList.i"
	@echo "... src/PotentialList.s"
	@echo "... src/PotentialStrategy.o"
	@echo "... src/PotentialStrategy.i"
	@echo "... src/PotentialStrategy.s"
	@echo "... src/SingleGaussPotential.o"
	@echo "... src/SingleGaussPotential.i"
	@echo "... src/SingleGaussPotential.s"
	@echo "... src/System.o"
	@echo "... src/System.i"
	@echo "... src/System.s"
	@echo "... src/TrapPotential.o"
	@echo "... src/TrapPotential.i"
	@echo "... src/TrapPotential.s"
	@echo "... src/Utils.o"
	@echo "... src/Utils.i"
	@echo "... src/Utils.s"
	@echo "... src/Variational.o"
	@echo "... src/Variational.i"
	@echo "... src/Variational.s"
	@echo "... src/fdcube.o"
	@echo "... src/fdcube.i"
	@echo "... src/fdcube.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system


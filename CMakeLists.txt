cmake_minimum_required(VERSION 3.0)
project(SVM_Project)

# Find libraries
find_package(Armadillo REQUIRED)

# Set GSL directory
set(GSL_DIR /home/frederik/gsl) 

# Bring the headers into the project (full or relative path)
include_directories(
	${ARMADILLO_INCLUDE_DIRS}
	${GSL_DIR}/include
        include
        )

# Set a variable with all the new flags
#set(GSL_FLAGS "-I/home/frederik/intel/mkl/include")
set(GSL_LINK_FLAGS "-L${GSL_DIR}/lib -lgsl -lgslcblas")

# Build custom library
file( GLOB LIB_SOURCES "src/*.cpp" )
file( GLOB LIB_HEADERS "include/*.h" )
add_library( SVMlib ${LIB_SOURCES} ${LIB_HEADERS} )


# Append the new flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Set directory for binary files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

# Build executables
file( GLOB APP_SOURCES mainfiles/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
    get_filename_component(sourcename ${sourcefile} NAME_WE)

    add_executable(
	${sourcename}
	${sourcefile}
	)

    # Make sure YourLib is linked to each app
    target_link_libraries( "${sourcename}" SVMlib ${ARMADILLO_LIBRARIES} ${GSL_LINK_FLAGS} )
endforeach( sourcefile ${APP_SOURCES} )

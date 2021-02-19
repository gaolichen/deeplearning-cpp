# check whether c++11 is supported.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

# example usage
#enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Werror")
#enable_cxx_compiler_flag_if_supported("-Wextra")
#enable_cxx_compiler_flag_if_supported("-pedantic")

# Your-external "mylib", add GLOBAL if the imported library is located in directories above the current.
#add_library(lrcalc SHARED IMPORTED)
# You can define two import-locations: one for debug and one for release.
#set_target_properties(lrcalc PROPERTIES IMPORTED_LOCATION ${LRCALC_ROOT}/lib/liblrcalc.so )

SET(DPL_LIB_SRCS
    activation.cpp
	common.cpp
	layer.cpp
	model.cpp
	node.cpp
	loss.cpp
	regularization.cpp
	datautil.cpp
	propagator.cpp
)

IF(WIN32)
	SET(DPL_LIB_SRCS 
		${DPL_LIB_SRCS}
		activation.h
        common.h
        layer.h
    	model.h
    	node.h
    	loss.h
    	regularization.h
    	datautil.h
    	propagator.h
	)
	source_group("Header Files" FILES *.h)
ENDIF(WIN32)

SET(DPL_MAIN_SRCS
	main.cpp)

#enable_testing()

#I like to keep test files in a separate source directory called test
file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cpp)

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_SOURCE_DIR} eigen)

#find_package(Boost 1.70)

#find_package(Boost 1.70 REQUIRED COMPONENTS
#             unit_test_framework)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_library (dpllib STATIC ${DPL_LIB_SRCS})
    add_executable (dpl ${DPL_MAIN_SRCS})
    target_link_libraries(dpl ${Boost_LIBRARIES} dpllib)

    # the test code depends on c++ 11. Support of c++0x is not enough.
    # if the compiler does not support c++11, do not build test code.
    if(NOT WIN32)
        add_executable(dpltest ${TEST_SRCS})
        target_link_libraries(dpltest ${Boost_LIBRARIES} dpllib)
        add_test(NAME bstest COMMAND bstest)
    endif()
endif()

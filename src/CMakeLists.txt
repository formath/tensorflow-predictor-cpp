cmake_minimum_required(VERSION 2.8)

include_directories(
	${PROJECT_SOURCE_DIR}/../tensorflow
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/proto
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/include
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/eigen
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
	${PROJECT_SOURCE_DIR}/include)

link_directories(
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/lib)

set(deep_ctr_LINKER_LIBS "")
list(APPEND deep_ctr_LINKER_LIBS libtensorflow-core.a)

# executable
set(exec_name "test.bin")
add_executable(${exec_name} "test.cc")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
target_link_libraries(${exec_name} ${deep_ctr_LINKER_LIBS})

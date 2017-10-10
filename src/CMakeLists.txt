cmake_minimum_required(VERSION 2.8)

include_directories(
	${PROJECT_SOURCE_DIR}/../tensorflow
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/proto
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/include
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/eigen
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
	${PROJECT_SOURCE_DIR}/include)

link_directories(
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/lib/ios_X86_64
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/lib
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/protobuf_ios/lib/iossim_x86_64/lib
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/nsync/builds/x86_64.ios.c++11)

set(DEEP_CTR_LINKER_LIBS "")
list(APPEND DEEP_CTR_LINKER_LIBS libtensorflow-core-x86_64.a libprotobuf.a libnsync.a pthread m z)

# executable
set(TEST "simple_model.bin")
add_executable(${TEST} "simple_model.cc")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
target_link_libraries(${TEST} ${DEEP_CTR_LINKER_LIBS})

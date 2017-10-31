cmake_minimum_required(VERSION 2.8)

include_directories(
	${PROJECT_SOURCE_DIR}/../tensorflow
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/proto
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/include
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/eigen
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/googletest/googletest/include
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/proto)

link_directories(
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/lib
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/gen/protobuf-host/lib)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()
if (LINUX)
	link_directories(
		${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11)
else()
	link_directories(
		${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/nsync/builds/default.maxos.c++11)
endif()

set(DEEP_CTR_LINKER_LIBS "")
list(APPEND DEEP_CTR_LINKER_LIBS
	libtensorflow-core.a
	libprotobuf.a
	${PROJECT_SOURCE_DIR}/../tensorflow/tensorflow/contrib/makefile/downloads/nsync/builds/default.macos.c++11/nsync.a
	pthread m z)
if (LINUX)
	list(APPEND DEEP_CTR_LINKER_LIBS dl)
endif()

# executable
set(SIMPLE_MODEL "simple_model.bin")
add_executable(${SIMPLE_MODEL} "simple_model.cc")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
target_link_libraries(${SIMPLE_MODEL} ${DEEP_CTR_LINKER_LIBS})

# executable
set(DEEP_MODEL "deep_model.bin")
add_executable(${DEEP_MODEL} "deep_model.cc" "../proto/dict.pb.cc")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
target_link_libraries(${DEEP_MODEL} ${DEEP_CTR_LINKER_LIBS})

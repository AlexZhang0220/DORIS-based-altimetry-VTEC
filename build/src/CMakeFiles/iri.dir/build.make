# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\CMake\bin\cmake.exe

# The command to remove a file.
RM = D:\CMake\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "D:\Master Thesis\iri2020\src\iri2020"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "D:\Master Thesis\build"

# Include any dependencies generated for this target.
include src/CMakeFiles/iri.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/iri.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/iri.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/iri.dir/flags.make

src/CMakeFiles/iri.dir/codegen:
.PHONY : src/CMakeFiles/iri.dir/codegen

src/CMakeFiles/iri.dir/irisub.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/irisub.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/irisub.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object src/CMakeFiles/iri.dir/irisub.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\irisub.for" -o CMakeFiles\iri.dir\irisub.for.obj

src/CMakeFiles/iri.dir/irisub.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/irisub.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\irisub.for" > CMakeFiles\iri.dir\irisub.for.i

src/CMakeFiles/iri.dir/irisub.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/irisub.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\irisub.for" -o CMakeFiles\iri.dir\irisub.for.s

src/CMakeFiles/iri.dir/irifun.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/irifun.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/irifun.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building Fortran object src/CMakeFiles/iri.dir/irifun.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\irifun.for" -o CMakeFiles\iri.dir\irifun.for.obj

src/CMakeFiles/iri.dir/irifun.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/irifun.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\irifun.for" > CMakeFiles\iri.dir\irifun.for.i

src/CMakeFiles/iri.dir/irifun.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/irifun.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\irifun.for" -o CMakeFiles\iri.dir\irifun.for.s

src/CMakeFiles/iri.dir/iritec.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/iritec.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/iritec.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building Fortran object src/CMakeFiles/iri.dir/iritec.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\iritec.for" -o CMakeFiles\iri.dir\iritec.for.obj

src/CMakeFiles/iri.dir/iritec.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/iritec.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\iritec.for" > CMakeFiles\iri.dir\iritec.for.i

src/CMakeFiles/iri.dir/iritec.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/iritec.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\iritec.for" -o CMakeFiles\iri.dir\iritec.for.s

src/CMakeFiles/iri.dir/iridreg.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/iridreg.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/iridreg.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building Fortran object src/CMakeFiles/iri.dir/iridreg.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\iridreg.for" -o CMakeFiles\iri.dir\iridreg.for.obj

src/CMakeFiles/iri.dir/iridreg.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/iridreg.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\iridreg.for" > CMakeFiles\iri.dir\iridreg.for.i

src/CMakeFiles/iri.dir/iridreg.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/iridreg.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\iridreg.for" -o CMakeFiles\iri.dir\iridreg.for.s

src/CMakeFiles/iri.dir/iriflip.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/iriflip.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/iriflip.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building Fortran object src/CMakeFiles/iri.dir/iriflip.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\iriflip.for" -o CMakeFiles\iri.dir\iriflip.for.obj

src/CMakeFiles/iri.dir/iriflip.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/iriflip.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\iriflip.for" > CMakeFiles\iri.dir\iriflip.for.i

src/CMakeFiles/iri.dir/iriflip.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/iriflip.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\iriflip.for" -o CMakeFiles\iri.dir\iriflip.for.s

src/CMakeFiles/iri.dir/igrf.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/igrf.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/igrf.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building Fortran object src/CMakeFiles/iri.dir/igrf.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\igrf.for" -o CMakeFiles\iri.dir\igrf.for.obj

src/CMakeFiles/iri.dir/igrf.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/igrf.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\igrf.for" > CMakeFiles\iri.dir\igrf.for.i

src/CMakeFiles/iri.dir/igrf.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/igrf.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\igrf.for" -o CMakeFiles\iri.dir\igrf.for.s

src/CMakeFiles/iri.dir/cira.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/cira.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/cira.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Building Fortran object src/CMakeFiles/iri.dir/cira.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\cira.for" -o CMakeFiles\iri.dir\cira.for.obj

src/CMakeFiles/iri.dir/cira.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/cira.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\cira.for" > CMakeFiles\iri.dir\cira.for.i

src/CMakeFiles/iri.dir/cira.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/cira.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\cira.for" -o CMakeFiles\iri.dir\cira.for.s

src/CMakeFiles/iri.dir/rocdrift.for.obj: src/CMakeFiles/iri.dir/flags.make
src/CMakeFiles/iri.dir/rocdrift.for.obj: D:/Master\ Thesis/iri2020/src/iri2020/src/rocdrift.for
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Master Thesis\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "Building Fortran object src/CMakeFiles/iri.dir/rocdrift.for.obj"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c "D:\Master Thesis\iri2020\src\iri2020\src\rocdrift.for" -o CMakeFiles\iri.dir\rocdrift.for.obj

src/CMakeFiles/iri.dir/rocdrift.for.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/iri.dir/rocdrift.for.i"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E "D:\Master Thesis\iri2020\src\iri2020\src\rocdrift.for" > CMakeFiles\iri.dir\rocdrift.for.i

src/CMakeFiles/iri.dir/rocdrift.for.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/iri.dir/rocdrift.for.s"
	cd /d D:\MASTER~1\build\src && D:\mingw64\bin\gfortran.exe $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S "D:\Master Thesis\iri2020\src\iri2020\src\rocdrift.for" -o CMakeFiles\iri.dir\rocdrift.for.s

iri: src/CMakeFiles/iri.dir/irisub.for.obj
iri: src/CMakeFiles/iri.dir/irifun.for.obj
iri: src/CMakeFiles/iri.dir/iritec.for.obj
iri: src/CMakeFiles/iri.dir/iridreg.for.obj
iri: src/CMakeFiles/iri.dir/iriflip.for.obj
iri: src/CMakeFiles/iri.dir/igrf.for.obj
iri: src/CMakeFiles/iri.dir/cira.for.obj
iri: src/CMakeFiles/iri.dir/rocdrift.for.obj
iri: src/CMakeFiles/iri.dir/build.make
.PHONY : iri

# Rule to build all files generated by this target.
src/CMakeFiles/iri.dir/build: iri
.PHONY : src/CMakeFiles/iri.dir/build

src/CMakeFiles/iri.dir/clean:
	cd /d D:\MASTER~1\build\src && $(CMAKE_COMMAND) -P CMakeFiles\iri.dir\cmake_clean.cmake
.PHONY : src/CMakeFiles/iri.dir/clean

src/CMakeFiles/iri.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "D:\Master Thesis\iri2020\src\iri2020" "D:\Master Thesis\iri2020\src\iri2020\src" "D:\Master Thesis\build" "D:\Master Thesis\build\src" "D:\Master Thesis\build\src\CMakeFiles\iri.dir\DependInfo.cmake" "--color=$(COLOR)"
.PHONY : src/CMakeFiles/iri.dir/depend


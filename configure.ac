### configure.ac					-*- Autoconf -*-

AC_PREREQ(2.69)

AC_INIT([xgboost],[1.7.11],[],[xgboost],[])

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

CXX17=`"${R_HOME}/bin/R" CMD config CXX17`
CXX17STD=`"${R_HOME}/bin/R" CMD config CXX17STD`
CXX="${CXX17} ${CXX17STD}"
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`

CC=`"${R_HOME}/bin/R" CMD config CC`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`

LDFLAGS=`"${R_HOME}/bin/R" CMD config LDFLAGS`
AC_LANG(C++)

DMLC_DEFS=""

AC_MSG_NOTICE([Checking if/where backtrace is available])
AC_CHECK_LIB([execinfo], [backtrace], [BACKTRACE_LIB=-lexecinfo], [BACKTRACE_LIB=''])
AS_IF([test -z "${BACKTRACE_LIB}"],
      dnl backtrace() might be unavailable (e.g., in musl libc)
      [AC_CHECK_FUNC(backtrace, [], [DMLC_DEFS="$DMLC_DEFS -DDMLC_LOG_STACK_TRACE=0"])])

AC_MSG_NOTICE([Checking whether fopen64 is available])
AC_CHECK_FUNC(fopen64, [], [DMLC_DEFS="$DMLC_DEFS -DDMLC_USE_FOPEN64=0"])

AC_MSG_NOTICE([Endian detection])
AC_MSG_CHECKING([endian])
AC_MSG_RESULT([])
AC_RUN_IFELSE([AC_LANG_PROGRAM([[#include <stdint.h>]], [[const uint16_t endianness = 256; return !!(*(const uint8_t *)&endianness);]])],
  [ENDIAN_FLAG="-DDMLC_CMAKE_LITTLE_ENDIAN=1"],
  [ENDIAN_FLAG="-DDMLC_CMAKE_LITTLE_ENDIAN=0"])

OPENMP_CXXFLAGS=""

if test `uname -s` = "Linux"
then
  OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CXXFLAGS)"
fi

if test `uname -s` = "Darwin"
then
  if command -v brew &> /dev/null
  then
    HOMEBREW_LIBOMP_PREFIX=`brew --prefix libomp`
  else
    # Homebrew not found
    HOMEBREW_LIBOMP_PREFIX=''
  fi
  OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include"
  OPENMP_LIB="-lomp -L${HOMEBREW_LIBOMP_PREFIX}/lib"
  ac_pkg_openmp=no
  AC_MSG_CHECKING([whether OpenMP will work in a package])
  AC_LANG_CONFTEST([AC_LANG_PROGRAM([[#include <omp.h>]], [[ return (omp_get_max_threads() <= 1); ]])])
  ${CXX} -o conftest conftest.cpp ${CPPFLAGS} ${LDFLAGS} ${OPENMP_LIB} ${OPENMP_CXXFLAGS} 2>/dev/null && ./conftest && ac_pkg_openmp=yes
  AC_MSG_RESULT([${ac_pkg_openmp}])
  if test "${ac_pkg_openmp}" = no; then
    OPENMP_CXXFLAGS=''
    OPENMP_LIB=''
    echo '*****************************************************************************************'
    echo '         OpenMP is unavailable on this Mac OSX system. Training speed may be suboptimal.'
    echo '         To use all CPU cores for training jobs, you should install OpenMP by running:'
    echo '             brew install libomp'
    echo '*****************************************************************************************'
  fi
fi

AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIB)
AC_SUBST(ENDIAN_FLAG)
AC_SUBST(DMLC_DEFS)
AC_SUBST(BACKTRACE_LIB)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

#ifndef KLT_CONFIG_H
#define KLT_CONFIG_H

/* OpenACC configuration */
#ifdef KLT_USE_OPENACC
  #define USE_OPENACC 1
  #ifdef KLT_OPENACC_NVC
    #define OPENACC_COMPILER_NVC 1
  #elif defined(KLT_OPENACC_GCC)
    #define OPENACC_COMPILER_GCC 1
  #endif
#else
  #define USE_OPENACC 0
#endif

#endif /* KLT_CONFIG_H */
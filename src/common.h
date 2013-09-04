#ifndef _GMM_COMMON_H_
#define _GMM_COMMON_H_

#include <stdio.h>

#define GMM_EXPORT __attribute__((__visibility__("default")))

#ifdef GMM_DEBUG
#define GMM_DPRINT(fmt, arg...) fprintf(stderr, "[gmm:debug] " fmt, ##arg)
#else
#define GMM_DPRINT(fmt, arg...)
#endif

#ifdef GMM_PROFILE
#define GMM_PRINT(fmt, arg...) fprintf(stderr, "[gmm:profile] " fmt, ##arg)
#else
#define GMM_PRINT(fmt, arg...)
#endif

#endif

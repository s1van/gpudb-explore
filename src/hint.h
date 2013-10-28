#ifndef _GMM_HINT_H_
#define _GMM_HINT_H_

// Read/write hints
#define HINT_READ		1
#define HINT_WRITE		2
#define HINT_PTRARRAY		4
#define HINT_PTAREAD		8
#define HINT_PTAWRITE		16
#define HINT_DEFAULT	(HINT_READ | HINT_WRITE)

// Kernel priority hints
#define PRIO_LOWEST		15
#define PRIO_DEFAULT	PRIO_LOWEST

#endif

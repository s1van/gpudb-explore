#ifndef _GMM_CLIENT_H_
#define _GMM_CLIENT_H_

// Functions exposed to client-side code to interact with global shared memory
int gmm_attach();
void gmm_detach();

long dev_memsize();
long dev_free_memsize();
long dev_free_memsize_signed();

#endif

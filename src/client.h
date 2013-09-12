#ifndef _GMM_CLIENT_H_
#define _GMM_CLIENT_H_

// Functions exposed to client-side code to interact with global shared memory
int gmm_attach();
void gmm_detach();

void begin_attach();
void end_attach();

long get_memsize();
long get_free_memsize();
long get_free_memsize_signed();

void update_attached(long delta);
void update_detachable(long delta);

#endif

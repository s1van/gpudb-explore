#ifndef _GMM_CLIENT_H_
#define _GMM_CLIENT_H_

#include <unistd.h>

// Functions exposed to client-side code to interact with global shared memory.
int client_attach();
void client_detach();

void begin_load();
void end_load();

long memsize();
long free_memsize();
long free_memsize2();

void update_attached(long delta);
void update_detachable(long delta);

int client_lru_detachable();
void client_unpin(int client);

int is_client_local(int client);

int getcid();
pid_t cidtopid(int cid);

#endif

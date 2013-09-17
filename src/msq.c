#include "msq.h"


mqd_t mqid_request = (mqd_t) -1;
pthread_t tid_request;
mqd_t mqid_notify = (mqd_t) -1;
//pthread_t tid_notify;		// we omit the notification thread for now

// For syncing with peer clients.
pthread_mutex_t mutx_msq;
pthread_cond_t cond_msq;

// The thread that receives and handles messages.
void *thread_msq_listener(void *arg)
{
	return NULL;
}

int msq_init()
{
	char qname[16];

	pthread_mutex_init(&mutx_msq);
	pthread_cond_init(&cond_msq);

	// Allocate request and notification message queues
	if (id >= 0 && id < pglobal->maxclients) {
		sprintf(qname, "/gr_%d", getpid());
		mqid_request = mq_open(qname, O_RDONLY | O_CREAT | O_EXCL, 0622, NULL);
		if (mqid_request == (mqd_t) -1) {
			GMM_DPRINT("failed to create the request queue: %s\n", \
					strerror(errno));
			free_client(id);
			return -1;
		}
		if (pthread_create(&tid_request, NULL, thread_request_listener,
				NULL) != 0) {
			mq_unlink(mqid_request);
			mq_close(mqid_request);
			mqid_request = (mqd_t) -1;
			free_client(id);
			return -1;
		}

		sprintf(qname, "/gn_%d", getpid());
		mqid_notify = mq_open(qname, O_RDONLY | O_CREAT | O_EXCL, 0622, NULL);
		if (mqid_notify == (mqd_t) -1) {
			GMM_DPRINT("failed to create the notification queue: %s\n", \
					strerror(errno));
			free_client(id);
			return -1;
		}
	}
}

void msq_fini()
{
	char qname[16];

	if (mqid_request != (mqd_t) -1) {
		sprintf(qname, "/gr_%d", pglobal->clients[id].pid);
		mq_unlink(qname);
		mq_close(mqid_request);
		mqid_request = (mqd_t) -1;
		// TODO: destroy the request listener thread
	}
	if (mqid_notify != (mqd_t)-1) {
		sprintf(qname, "/gn_%d", pglobal->clients[id].pid);
		mq_unlink(qname);
		mq_close(mqid_notify);
		mqid_notify = (mqd_t)-1;
	}
}

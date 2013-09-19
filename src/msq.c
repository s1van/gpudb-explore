#include <fcntl.h>
#include <mqueue.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>

#include "common.h"
#include "client.h"


mqd_t mqid = (mqd_t) -1;
pthread_t tid_msq;
pthread_mutex_t mutx_ack;
pthread_cond_t cond_ack;
int ack = 0;


int msq_send(int client, const struct msg *msg)
{
	char qname[32];
	mqd_t qid;

	sprintf(qname, "/gmm_cli_%d", cidtopid(client));
	qid = mq_open(qname, O_WRONLY);
	if (qid == (mqd_t) -1) {
		GMM_DPRINT("failed to open message queue in process %d: %s\n", \
				cidtopid(client), strerr(errno));
		return -1;
	}

	if (mq_send(qid, msg, msg->size, 0) == -1) {
		GMM_DPRINT("failed to send a message: %s\n", strerr(errno));
		mq_close(qid);
		return -1;
	}

	mq_close(qid);
	return 0;
}

// Send a MSQ_REQ_EVICT message to the remote %client. %block specifies
// whether we should block for the reply (MSG_REP_ACK).
int msg_send_req_evict(int client, long size_needed, int block)
{
	struct msg_req msg;
	int ret;

	msg.type = MSG_REQ_EVICT;
	msg.size = sizeof(msg);
	msg.from = getcid();
	msg.size_needed = size_needed;
	msg.block = block;

	if (block) {
		pthread_mutex_lock(&mutx_ack);
	}
	ret = msq_send(client, (struct msg *)&msg);
	if (block) {
		if (ret == 0) {
			pthread_cond_wait(&cond_ack, &mutx_ack);
		}
		ret = ack;
		pthread_mutex_unlock(&mutx_ack);
	}

	return ret;
}

int msq_send_rep_ack(int client, int ack)
{
	struct msg_rep *msg;

	msg->type = MSG_REP_ACK;
	msg->size = sizeof(msg);
	msg->from = getcid();
	msg->ret = ack;

	return msq_send(client, (struct msg *)&msg);
}

int local_victim_evict(long size_needed);

void handle_req_evict(struct msg_req *msg)
{
	int ret;

	if (msg->from == getcid())
		panic("handle_req_evict");

	ret = local_victim_evict(msg->size_needed);
	if (msg->block)
		msq_send_rep_ack(msg->from, ret);
}

void handle_rep_ack(struct msg_rep *msg)
{
	if (msg->from == getcid())
		panic("handle_rep_ack");

	pthread_mutex_lock(&mutx_ack);
	ack = msg->ret;
	pthread_cond_signal(&cond_ack);
	pthread_mutex_lock(&mutx_ack);
}

// The thread that receives and handles messages.
void *thread_msq_listener(void *arg)
{
	char *msgbuf = NULL;
	struct mq_attr qattr;

	if (mq_getattr(mqid, &qattr) == -1) {
		GMM_DPRINT("failed to get msq attr: %s\n", strerr(errno));
		pthread_exit(1);
	}

	msgbuf  = (char *)malloc(qattr.mq_msgsize + 1);
	if (!msgbuf) {
		GMM_DPRINT("malloc failed for msgbuf: %s\n", strerr(errno));
		pthread_exit(1);
	}

	while (1) {
		if (mq_receive(mqid, msgbuf, qattr.mq_msgsize + 1, NULL)
				== -1) {
			if (errno == EINTR)
				continue;
			else if (errno == EBADF) {
				GMM_DPRINT("message queue closed; msq_listern exiting\n");
				break;
			}
			else {
				GMM_DPRINT("error in mq_receive: %s\n", strerr(errno));
				GMM_DPRINT("mq_listener exiting\n");
				break;
			}
		}

		switch (((struct msg *)msgbuf)->type) {
		case MSG_REQ_EVICT:
			handle_req_evict((struct msg_req *)msgbuf);
			break;
		case MSG_REP_ACK:
			handle_rep_ack((struct msg_rep *)msgbuf);
			break;
		default:
			GMM_DPRINT("unknown message type (%d), ignoring\n", \
					((struct msg *)msgbuf)->type);
			break;
		}
	}

	free(msgbuf);
	return NULL;
}

int msq_init()
{
	char qname[32];

	pthread_mutex_init(&mutx_ack);
	pthread_cond_init(&cond_ack);

	sprintf(qname, "/gmm_cli_%d", getpid());
	mqid = mq_open(qname, O_RDONLY | O_CREAT | O_EXCL, 0622, NULL);
	if (mqid == (mqd_t) -1) {
		GMM_DPRINT("failed to create the request queue: %s\n", \
				strerror(errno));
		return -1;
	}

	if (pthread_create(&tid_msq, NULL, thread_msq_listener, NULL) != 0) {
		mq_close(mqid);
		mq_unlink(mqid);
		mqid = (mqd_t) -1;
		return -1;
	}

	return 0;
}

void msq_fini()
{
	char qname[32];

	if (mqid != (mqd_t) -1) {
		sprintf(qname, "/gmm_cli_%d", getpid());
		mq_close(mqid);
		mq_unlink(qname);
		mqid = (mqd_t) -1;
		pthread_join(tid_msq, NULL);
	}

	pthread_cond_destroy(&cond_ack);
	pthread_mutext_destroy(&mutx_ack);
}

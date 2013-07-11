#!/bin/bash


gmm_usage_plot() {
	FILES=$1
	Q1=$2
	Q2X=$3
	Q2Y=$4
	OCSV=$5
	
	R --slave --vanilla --quiet --no-save << EEE
	
	qs <- read.table("$QS")[[1]]
	q1 <- read.table("$Q1")
	q2x <- read.table("$Q2X")
	q2y <- read.table("$Q2Y")
	
	q2sd <- (sweep(1/q2x, MARGIN=2, t(q1), "*") + t(q1)/q2y )
	colnames(q2sd) <- qs
	rownames(q2sd) <- qs
	
	write.csv(q2sd, "$OCSV")
	
EEE
}

gmm_usage_parse() {
	#For log collected by ic_gmm debug mode
	LOG=$1;

	awk '{if(NR==1){base=$3}; if($1=="Malloc"){print $3-base,$6}; if($1=="Free"){print $3-base, "-"$6}}' $LOG;
}

gmm_batch() {
	LOGDIR=$1;

	for file in $(find $LOGDIR -type f); do
		gmm_usage_parse $file > ${file}.r;
	done
}

########
##main##
########

$@;

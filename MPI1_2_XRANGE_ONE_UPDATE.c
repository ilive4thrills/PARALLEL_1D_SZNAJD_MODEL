/* https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_array.c, TO RUN EVERYTHING, use mpirun, and to compile use mpicc */
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>


/* Here is what the command line implementation should look like: */
/* mpiexec -n 2 ./mpi2 TOTALSIZE P_VALUE NUM_OF_RUNS */
/*                  [0]    [1]      [2]      [3]         */

#define  XSIZE          100
#define  MASTER		0
#define  UP             1
#define  DOWN           0
#define  UNOCCUPIED     -1
#define  UNLOCKED       0
#define  LOCKED         1    
#define  INDEX_LOW(id,p,n) ((id) * (n)/(p))
#define  INDEX_HIGH(id,p,n) (INDEX_LOW((id)+1,p,n) - 1)                   /* id == process rank */
#define  BLOCK_SIZE(id,p,n) (INDEX_LOW((id) + 1) - INDEX_LOW(id))                      /* n == number of array elements */
#define  BLOCK_OWNER(index,p,n) (((p)*((index) +1 )-1)(n))               /* p == number of processes */
                                                /* index == array index */
void init_array(int sublattice[],int sublatticesize);  // OKAY FOR NEW OPERATION
void randomization_init(int sublattice[], int * c, int * Number_up, int * Number_down, int sublatticesize);  // OKAY FOR NEW OPERATION
int random_num_interval (int sublatticesize);  /*actual implementation of the random integer number generator function*/ // OKAY FOR NEW OPERATION
void evolve_sublattice(int sublattice[], int sublatticesize,int *flag);  // I DON'T THINK THIS IS EVEN BEING USED
int boundary_conditions(int sublatticesize, int rsn);  // I DON'T THINK THIS IS EVEN BEING USED
void neighbor_find_update(int sublattice[], int array[], int * number_up, int * number_down, double p);
void update_neighbors(int lattice[], int array[], double p_cA, double p_cB, double p);
int global_assignments(int global_ss, int * global_ss2, int * global_ln, int * global_rn, int latticesize);
void create_and_open_files(FILE * output, FILE * output_MCS, FILE * output_m, char * latticesize, char * p, char * NOR);
void concatenate_strings(char * string, char * message, char * latticesize, char * p, char * NOR);
long random_at_most(long max);

void main (int argc, char *argv[])
{
if (argc < 4 ) {
	printf("You forgot to include one or more arguments.\n");
}

struct rlimit rlim = { RLIM_INFINITY, RLIM_INFINITY };
if ( setrlimit(RLIMIT_STACK, &rlim) == -1 ) {                       /* Increase stack size */
	perror("setrlimit error");
	exit(1);
} 

int random_time = time(NULL);    /*seed that random integer with the current time, etc.*/
srand(random_time);             /*continuation of seeding the pseudorandom number generator*/
/* tracking variables for simulation quantities */
clock_t start = clock();
char    buffer[1024];
int     numtasks;
int     reminder;
int     sublatticesize;
int     latticesize;
double  p;
int     NOR;
int * 	lattice;
int     taskid;
int     i;
int     j;
int     x;
int     c; 
int     t;
int     k;
int     one = 1;
int     proc;
int     number_up;
int     number_down;
int 	low_neighbor;
int 	high_neighbor;
int     number_up_sum;
int     localstep;
int     colltd_step;
int     colltd_numup_sum;
int     colltd_numdn;
int     colltd_mcs;
int     MCS_count;
int     flag;
int     flag1;
int     rsn;
int     global_rsn;
int     global_ss2;
int     global_rn;
int     global_ln;
int 	c1;
int     remainder = 0;
double  x_value;
double  m;
float   mysum;
float   sum;
double  globetime;
int     begin_x = 0;
int     end_x = 0;
FILE *  output = 0;               /*output file pointer for all E(x,L) information*/
FILE *  output_MCS = 0;           /*output file pointer for all MCS information*/
FILE *  output_m = 0;             /*output file pointer for all m(t) information */
FILE *  out1 = 0;
FILE *  out2 = 0;
int     INDEX_LOW;
int     INDEX_HIGH;
int 	array[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};   // various numbers that each process should know, such as neighbors, numup, numdown, etc. 

/***** MPI Initializations *****/
MPI_Init(&argc, &argv);                /* initialize the MPI context */
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);         /* use MPI_COMM_WORLD to determine number of tasks (processors) */
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);

latticesize = atoi(argv[1]);
sublatticesize = atoi(argv[1])/numtasks;
p = atof(argv[2]);
NOR = atoi(argv[3]);
begin_x = atoi(argv[4]);
end_x = atoi(argv[5]);

//argv[0] == ./MPI1_2_XRANGE_ONE_UPDATE
//argv[1] == LATTICESIZE
//argv[2] == P-VALUE
//argv[3] == NOR
//argv[4] == BEGIN_X
//argv[5] == END_X

int orig_NOR = NOR;

if((NOR % numtasks) == 0) {
	  NOR = NOR/numtasks;   // all processes get an equal number of iterations
} 
if ((orig_NOR % numtasks) != 0) {
    remainder = NOR % numtasks;    // how many leftover iterations there are 
    NOR = (NOR - remainder)/numtasks; // all processes get an equal base number of iterations
    if (taskid == 0) {           // have root process disperse the random sites
	NOR = NOR + remainder;   // process 0 takes the hit and gets the extra processes
    }
}  //some c_locals should be higher than they were before at this point.


int test = 0;

//for(test = 0; test < argc; test++) {
//	printf("argv[%d] == %s\n", test, argv[test]);
//}

MPI_Barrier(MPI_COMM_WORLD); // synchronize, make sure each process has the right number of iterations
lattice = (int *)malloc(latticesize*sizeof(int)); 

char outstr[200];
time_t time_data;
struct tm *tmp;

time_data = time(NULL);
tmp = localtime(&time_data);

if (tmp == NULL) {
    perror("localtime");
    exit(EXIT_FAILURE);
}

if(strftime(outstr, sizeof(outstr), "%m%d%y", tmp) == 0) {
     fprintf(stderr, "strftime returned 0");
     exit(EXIT_FAILURE);
}

//printf("Result string is \"%s\"\n", outstr);

if(taskid == 0) {
	sprintf(buffer, "N%d_%f_%d_%d_%d_P%d_mt_%s.dat", latticesize, p, orig_NOR, begin_x, end_x, numtasks, outstr);
	output_m = fopen(buffer, "a");
	sprintf(buffer, "N%d_%f_%d_%d_%d_P%d_MCS_%s.dat", latticesize, p, orig_NOR, begin_x, end_x, numtasks, outstr);
	output_MCS = fopen(buffer, "a");	
}
printf("Number of runs for this process (%d) is %d\n", taskid, NOR);

array[0] = taskid;
array[1] = numtasks;
array[2] = sublatticesize;
array[3] = 0;   // up
array[4] = 0;    //down
array[5] = 0;    //ln
array[6] = 0;    // ss/ss2
array[7] = 0;    //rn
array[8] = low_neighbor;
array[9] = high_neighbor;
array[10] = latticesize;
array[11] = 0;       // rsn
//array[12] = INDEX_LOW;
//array[13] = INDEX_HIGH;

printf("begin_x = %d and end_x = %d\n", begin_x, end_x);
for (x = begin_x; x < (end_x + 1); x++) {
	x_value = ((double)x)/((double)XSIZE);
	c = x_value*latticesize;   // c is the total concentration c = (0.65*latticesize) = 360 sites
	number_up_sum = 0;
	for (i = 0; i < NOR; i++) {
		number_up = 0;
		number_down = 0;
		MCS_count = 0; 
		localstep = 0;
		c1 = c;
		init_array(lattice, latticesize);
		randomization_init(lattice, &c1, &number_up, &number_down, latticesize);   
		do {
			neighbor_find_update(lattice, array, &number_up, &number_down, p);
			localstep += 1;
			number_up = 0, number_down = 0;
			for (k = 0; k < latticesize; k++) {
				if (lattice[k] == UP) number_up += 1;
			}
			number_down = latticesize - number_up;
			if ((taskid == 0) && ((localstep % latticesize) == 0)) { 
				m = ((double)(number_up - number_down))/((double)latticesize);
				if (output_m == NULL) fprintf(stdout,"output_m file pointer NULL\n");
				MCS_count = localstep / latticesize;
				fprintf(output_m, "%d %f %d %f %d\n", taskid, x_value, i, m, MCS_count);
			}
			if ((taskid == 0)) {   // make sure we get the exact MCS required to get the system to consensus
				m = ((double)(number_up - number_down))/((double)latticesize);
				if (fabs(m) == 1) {             // if we have reached consensus, we want to note the MC step that consensus occurred at 
					fprintf(output_m, "%d %f %d %f %f\n", taskid, x_value, i, m, (double)localstep/(double)latticesize);
				}
			}
		} while ((number_up < latticesize) && (number_up > 0));
		if ((taskid == 0) && (x_value == 0.5)) {
			if (output_MCS == NULL) fprintf(stdout,"output_MCS file pointer NULL\n");
			fprintf(output_MCS,"%f %d %d\n",x_value, i, MCS_count);
		}
		if (number_up == latticesize) {
			number_up_sum += 1;
		}
	}
	fprintf(stdout, "%d %f %d %d\n", taskid, x_value, number_up_sum, NOR);  
}
/*closure stuff here. */
clock_t finish = clock();
double simtime = (((double) (finish - start)) / (CLOCKS_PER_SEC));   /*compute how long the simulation took in HMSec., print that to screen*/
//MPI_Reduce(&simtime, &globetime, 1, MPI_DOUBLE, MPI_SUM, MASTER,MPI_COMM_WORLD);
fprintf(stdout, "simtime is %f in process %d\n", simtime, taskid);

free(lattice);

printf("in process %d and we just freed LATTICE.\n", taskid);
if (taskid == 0) {
	close(output_MCS);
	close(output_m);
}

MPI_Finalize();
}  
/* end of main */

void neighbor_find_update(int lattice[], int array[], int * number_up, int * number_down, double p)
{ 
	int latticesize = array[10];
	int ss = random_at_most((latticesize - 1)); // pick random site within lattice  //int global_ss = rsn + (taskid)*(sublatticesize);
	int ss2 = 0, rn = 0, ln = 0;
	double p_cA = 0;
	double p_cB = 0;
	p_cA = ((double)rand()) / ((double)RAND_MAX);
	p_cB = ((double)rand()) / ((double)RAND_MAX);
	global_assignments(ss, &ss2, &ln, &rn,latticesize); // determine what the four sites of interest are. PROBLEM!!!!!
	
	array[5] = ln;    //ln
	array[6] = ss;    // ss, not ss2 (THIS IS the global version of rsn)
	array[7] = rn;    //rn

	if (lattice[ss] == lattice[ss2]) {
	  update_neighbors(lattice, array, p_cA, p_cB, p);
	}
}

int global_assignments(int ss, int * ss2, int * ln, int * rn, int latticesize)  // Note This!!!! RSN is equivalent to SS!!!
{
	if (ss == 0) {                            // BCs for low end of the sublattice
		*ln = latticesize - 1;
		*ss2 = 1;
		*rn = 2;
	}
	else if (ss == (latticesize - 1)) {               // these XXXXXXXXX bottom control statements are for the ends of the large lattice
		*ln = ss - 1;
		*ss2 = 0;
		*rn = 1;
	}
	else if (ss == (latticesize - 2)) {
		*ln = ss - 1;
		*ss2 = ss + 1;
		*rn = 0;                                  // normal neighbor configuration, but we will need to incorporate BCs for updating
	} else {                                /*Standard Bookend Condition: [x x x x ln rsn ss2 rn x x x x x x]*/
	  *ss2 = ss + 1;
	  *ln = ss - 1;
	  *rn = ss + 2;
	}
}


long random_at_most(long max) 
{
	unsigned long num_bins = (unsigned long) max + 1, num_rand = (unsigned long) RAND_MAX + 1,
		      bin_size = num_rand / num_bins, defect = num_rand % num_bins;
	long x;
	do {
		x = random();
	} while (num_rand - defect <= (unsigned long)x);
return x/bin_size;
}

void init_array(int sublattice[],int sublatticesize)
{
  int j = 0;
  for (j = 0; j < sublatticesize; j++) {
    sublattice[j] = DOWN;
  }
}

void print_sublattice(int sublattice[], int sublatticesize, FILE * outn)
{
	//print_with_indent(taskid*sublatticesize);
	int j = 0;
	for(j = 0; j < sublatticesize; j++) {
		fprintf(outn, "%d",sublattice[j]);
	}
	fprintf(outn,"\n");
}
void print_lattice(int sublattice[], int arraysize)
{
	int taskid;
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	int j = 0;
	for( j = 0; j < arraysize; j++) {
		printf(" {%d %d } ", sublattice[j], taskid);
	}
	printf("\n");
}
void print_with_indent(int indent)    /* from jk_ */
{
    printf("%*s", indent, " "); 
}

void randomization_init(int sublattice[], int * c, int * Number_up, int * Number_down, int sublatticesize)
{
	int max_index = sublatticesize - 1;
	int site = 0;
	while (*c > 0) {
        	site = random_at_most(max_index);  /*select a random site to make "up", within the array*/
		if (sublattice[site] == DOWN) {  /*make sure that site is "down"*/
        		sublattice[site] = UP;   /*make the "down" site "up"*/
	       		*c = *c - 1;                         /*decrement c, as we have decreased the amount of sites we need to make "up" by 1*/
        		*Number_up = *Number_up + 1;       /*increase by 1 the amount of "up" sites*/
	       		*Number_down = sublatticesize - (*Number_up);   /*number_down = Num_of_sites - Number_up*/
        	}
    	}
}
// SPLIT RUNS AMONGST ARBITRARY NUMBER OF PROCESSES!!!!
void update_neighbors(int lattice[], int array[], double p_cA, double p_cB, double p)
{         // diagram of the four-site panel [ln][rsn/ss][ss2][rn]
	int ln = array[5];
	int ss = array[6];
	int rn = array[7];
	int latticesize = array[10];
	int rand_lr_site = rand() % latticesize;      // Pick one (1) random long-range site to update
	double pair_longorshortup_prob = ((double)rand()) / ((double)RAND_MAX);	//range is [0, 1] % determine what the P is for our pair (are we updating one long-range site or one short-range site??)
	double coin_flip = rand() % 2;                  // if we are updating a short-range site, pick left or right randomly
        
	//printf("lr_A = %d, lr_B = %d, lattice = %d, p_cA = %f, p_cB = %f\n", lr_A, lr_B, latticesize, p_cA, p_cB);
	if (pair_longorshortup_prob < p){ // pair randomly updates one long-range neighbor 
		lattice[rand_lr_site] = lattice[ss];
	}
	if (pair_longorshortup_prob > p) {     // pair flips a coin to update the immediate right or left neighbor
		if (coin_flip == 0) // update left
		lattice[ln] = lattice[ss];
		if (coin_flip == 1) // update right
		lattice[rn] = lattice[ss];	// no probability calculation, bcz p = 0
	}
}

void create_and_open_files(FILE * output, FILE * output_MCS, FILE * output_m, char * latticesize, char * p, char * NOR)
{
	char * three_strings = malloc(strlen("SM1_Data/N=") + 2*strlen(latticesize) + strlen("_Runs/") + strlen(p) + strlen(NOR) + 3 + strlen("_MCS_930.dat"));
	char * three_strings2 = malloc(strlen("SM1_Data/N=") + 2*strlen(latticesize) + strlen("_Runs/") + strlen(p) + strlen(NOR) + 3 + strlen("_NC_930.dat"));
	char * three_strings3 = malloc(strlen("SM1_Data/N=") + 2*strlen(latticesize) + strlen("_Runs/") + strlen(p) + strlen(NOR) + 3 + strlen("_m(t)_930.dat"));
	
	concatenate_strings(three_strings, "_MCS_930.dat", latticesize, p, NOR);
	concatenate_strings(three_strings2, "_NC_930.dat", latticesize, p, NOR);
	concatenate_strings(three_strings3, "_m(t)_930.dat", latticesize, p, NOR);
	
	output_MCS = fopen(three_strings, "a+");   /*naming the MCS information file*/
	output = fopen(three_strings2, "a+");      /*naming the E(x,L) information file.*/
	output_m = fopen(three_strings3, "a+");
	
	free(three_strings);
	free(three_strings2);
	free(three_strings3);
	if ((output_m == NULL) || (output_MCS == NULL) || (output == NULL)) printf("error starts here\n");
}

void concatenate_strings(char * string, char * message, char * latticesize, char * p, char * NOR) 
{
	strcpy(string, "SM1_Data/N=");
	strcat(string, latticesize);
	strcat(string, "_Runs/"); 
	strcat(string, latticesize);
	strcat(string, "_");
	strcat(string, p);
	strcat(string, "_");
	strcat(string, NOR);
	strcat(string, message);
}



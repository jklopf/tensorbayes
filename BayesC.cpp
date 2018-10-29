#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#include <random>
#include <map>
#include <string>
#include <iomanip>

#include <unistd.h>
#include <string>
#include <algorithm>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/random.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>
#include <boost/program_options.hpp>
#include <iterator>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::Map;
using Eigen::Upper;
typedef Map<MatrixXd> MapMatd;

boost::random::mt19937 gen(time(0));

//distributions
double runif(double lower, double higher)
{
	boost::random::uniform_real_distribution<> dist(lower, higher);
	return dist(gen);
}

double rnorm(double mean, double sd)
{
	boost::random::normal_distribution<> dist(mean, sd);
	return dist(gen);
}


double rbeta(double alpha, double beta)
{

	boost::math::beta_distribution<> dist(alpha, beta);
	double q = quantile(dist, runif(0,1));

	return(q);
}

double rinvchisq(double df, double scale)
{

	boost::math::inverse_chi_squared_distribution<> dist(df, scale);
	double q = quantile(dist, runif(0,1));

	return(q);
}
int rbernoulli(double p)
{
	std::bernoulli_distribution dist(p);
	return dist(gen);
}

//sampling functions
double sample_mu(int N, double Sigma2_e,const VectorXd& Y,const MatrixXd& X,const VectorXd& beta)
{
	double mean=((Y-X*beta).sum())/N;
	double sd=sqrt(Sigma2_e/N);
	double mu=rnorm(mean,sd);
	return(mu);
}

//sample variance of beta
double sample_sigma2_b(const VectorXd& beta,int NZ,double v0B,double s0B){
	double df=v0B+NZ;
	double scale=(beta.squaredNorm()*NZ+v0B*s0B)/(v0B+NZ);
	//cout<<NZ<<"\t"<<beta.squaredNorm()<<"\t"<<df<<"\t"<<scale<<"\t"<<endl;
	double psi2=rinvchisq(df, scale);
	return(psi2);
}

//sample error variance of Y
double sample_sigma2_e(int N,const VectorXd& epsilon,double v0E,double s0E){
	double sigma2=rinvchisq(v0E+N, (epsilon.squaredNorm()+v0E*s0E)/(v0E+N));
	return(sigma2);
}

//sample mixture weight
double sample_w(int M,int NZ){
	double w=rbeta(1+NZ,1+(M-NZ));
	return(w);
}


void ReadFromFile(std::vector<double> &x, const std::string &file_name)
{
	std::ifstream read_file(file_name);
	assert(read_file.is_open());

	std::copy(std::istream_iterator<double>(read_file), std::istream_iterator<double>(),
			std::back_inserter(x));

	read_file.close();
}

int main(int argc, char *argv[])
{

	po::options_description desc("Options");
	desc.add_options()
		("M", po::value<int>()->required(), "No. of simulated markers")
		("N", po::value<int>()->required(), "No. of simulated individuals")
		("iter", po::value<int>()->default_value(5000), "No. of Gibbs iterations")
		("pNZ", po::value<double>()->default_value(0.5), "Proportion nonzero")
		("input", po::value<std::string>()->default_value("none"),"Input filename")
		("out", po::value<std::string>()->default_value("BayesC_out"),"Output filename")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,desc),vm);
	po::notify(vm);

	int M=vm["M"].as<int>();
	int N=vm["N"].as<int>();
	int iter=vm["iter"].as<int>();
	string input=vm["input"].as<string>();
	string output=vm["out"].as<string>();

	MatrixXd X(N,M);
	VectorXd Y(N);

	//beta coefficients
	VectorXd beta_true(M);
	beta_true.setZero();

	int i,j,k,l,m=0;

	//Was an input matrix given?

	if (input!="none"){ //Either read input tables for X and Y
		ifstream f1(input+".X");
		//f1 >> m >> n;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				f1 >> X(i,j);
				//cout<<X(i,j)<<endl;
			}
		}
		f1.close();
		cout<<"finished reading matrix X!"<<endl;

		std::vector<double> Y_in;
		ReadFromFile(Y_in, input+".Y");
		double* ptr_Y = &Y_in[0];
		Eigen::Map<Eigen::VectorXd> Y1(ptr_Y, Y_in.size());
		cout<<"finished reading vector Y!"<<endl;
		if(Y_in.size()!=N){cout<<"input Y vector size doesnt much the size indicated in the command line"<<endl;return 0;}
		Y=Y1;
		Y_in.clear();


	}else //or simulate
	{
		double pNZ=vm["pNZ"].as<double>();
		double sigmaY_true=1;
		double sigmab_true=1;
		int MT=pNZ*M;

		//Fill Genotype matrix
		for (i=0;i<N;i++){
			for (j=0;j<M;j++){
				X(i,j)=rnorm(0,1);
			}
		}
		for (i=0;i<MT;i++){
			beta_true[i]=rnorm(0,sigmab_true);
		}

		//error
		VectorXd error(N);
		for (i=0;i<N;i++){
			error[i]=rnorm(0,sigmaY_true);
		}

		//construct phenotypes
		Y=X*beta_true;
		Y+=error;
	}

	//normalize
	RowVectorXd mean = X.colwise().mean();
	RowVectorXd sd = ((X.rowwise() - mean).array().square().colwise().sum() / (X.rows() - 1)).sqrt();
	X = (X.rowwise() - mean).array().rowwise() / sd.array();


	double Emu=0;
	VectorXd vEmu(N);
	vEmu.setOnes();

	VectorXd Ebeta(M);
	Ebeta.setZero();
	VectorXd ny(M);
	ny.setZero();
	double Ew=0.5;
	//residual error
	VectorXd epsilon(N);

	epsilon=Y-X*Ebeta-vEmu*Emu;

	std::vector<int> markerI;
	for (int i=0; i<M; ++i) {
		markerI.push_back(i);
	}
	int marker=0;

	//non-zero variable NZ
	int NZ=0;

	double Sigma2_e=epsilon.squaredNorm()/(N*0.5);
	double Sigma2_b=rbeta(1,1);

	//Standard parameterization of hyperpriors for variances
	//double v0E=0.001,s0E=0.001,v0B=0.001,s0B=0.001;


	// Alternative parameterization of hyperpriors for variances
	double v0E=4,v0B=4;
	double s0B=((v0B-2)/v0B)*Sigma2_b;
	double s0E=((v0E-2)/v0E)*Sigma2_e;


	//pre-computed elements for calculations
	VectorXd el1(M);
	for (int i=0; i<M; ++i) {
		el1[i]=X.col(i).transpose()*X.col(i);
	}

	std::ofstream ofs;
	ofs.open(output+"_estimates.txt");
	for (int i=0; i<M; ++i) {
		ofs << "beta_" <<i<< ' ';
	}
	for (int i=0; i<M; ++i) {
		ofs << "incl_" <<i<< ' ';
	}
	ofs << "Ew" << " ";
	ofs << "Sigma2_b" << " ";
	ofs << "Sigma2_e" << " ";
	ofs << "\n";
	ofs.close();

	//begin GIBBS sampling iterations

	ofs.open (output+"_estimates.txt", std::ios_base::app);
	for (i=0;i<iter;i++){

		Emu=sample_mu(N,Sigma2_e,Y,X,Ebeta);

		//sample effects and probabilities jointly
		std::random_shuffle(markerI.begin(), markerI.end());
		for (j=0;j<M;j++){
			marker=markerI[j];

			epsilon=epsilon+X.col(marker)*Ebeta[marker];

			double Cj=el1[marker]+Sigma2_e/Sigma2_b; //adjusted variance
			double rj=X.col(marker).transpose()*epsilon; // mean



			double ratio=(((exp(-(pow(rj,2))/(2*Cj*Sigma2_e))*sqrt((Sigma2_b*Cj)/Sigma2_e))));
			ratio=Ew/(Ew+ratio*(1-Ew));
			ny[marker]=rbernoulli(ratio);

			if (ny[marker]==0){
				Ebeta[marker]=0;
			}
			else if (ny[marker]==1){
				Ebeta[marker]=rnorm(rj/Cj,Sigma2_e/Cj);
			}

			epsilon=epsilon-X.col(marker)*Ebeta[marker];

		}
		for (j=0;j<M;j++){
			ofs << Ebeta[j] << " ";
		}
		for (j=0;j<M;j++){
			ofs << ny[j] << " ";
		}
		NZ=ny.sum();
		//cout<<NZ<<endl;

		Ew=sample_w(M,NZ);
		epsilon=Y-X*Ebeta-vEmu*Emu;

		Sigma2_b=sample_sigma2_b(Ebeta,NZ,v0B,s0B);
		Sigma2_e=sample_sigma2_e(N,epsilon,v0E,s0E);

		ofs << Ew << " ";
		ofs << Sigma2_b << " ";
		ofs << Sigma2_e << " ";
		ofs << "\n";

	}
	ofs.close();

if (input=="none"){
	//write to files
	ofstream myfile1;
	myfile1.open (output+"_simulated_Y.txt");
	for (i=0;i<N;i++){
		myfile1 << Y[i] << ' ';
	}
	myfile1 << endl;
	myfile1.close();

	ofstream myfile2;
	myfile2.open (output+"_simulated_X.txt");
	for (i=0;i<N;i++){
		for (j=0;j<M;j++){
			myfile2<<X(i,j)<< ' ';
		}
		myfile2<<endl;
	}
	myfile2.close();

	ofstream myfile3;
	myfile3.open (output+"_simulated_betatrue.txt");
	myfile3 << beta_true << ' ';
	myfile3.close();
}

	return 0;
}

#include <Rcpp.h>
#include <vector>
#include <deque>
#include <utility>

// [[Rcpp::export]]
Rcpp::NumericMatrix slidingWindowMin(Rcpp::NumericVector cords, Rcpp::NumericVector values, double winsize)
{
	Rcpp::NumericMatrix result(values.size(), 3);
	colnames(result) = Rcpp::CharacterVector::create("value", "time", "idx");

	double wsize = winsize/2.;

	// pair<pair<double, double>, int> represents the tuple (values[i], cords[i], i)
	std::deque< std::pair<std::pair<double, double>, int> > window;
	
	int old_end=0;

	for (int i = 0; i < values.size(); i++) {
		int j=old_end;
		while(j<values.size() && cords[j]<=cords[i]+wsize)
		{
			while (!window.empty() && (window.back()).first.first >= values[j])
				window.pop_back();

			window.push_back(std::make_pair(std::make_pair(values[j], cords[j]), j));

			j++;
		}

		old_end = j;

		while((window.front()).first.second < cords[i] - wsize)
			window.pop_front();
		
		result(i, 0) = (window.front()).first.first;
		result(i, 1) = (window.front()).first.second;
		result(i, 2) = (window.front()).second+1;
	}

	return result;
}
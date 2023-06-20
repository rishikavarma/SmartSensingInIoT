#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace std;
double findMean(vector<double>a, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
 
    return (double)sum / (double)n;
}

double variance(vector<double> a, int n)
{
    // Compute mean (average of elements)
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    double mean = (double)sum /
                  (double)n;
 
    // Compute sum squared
    // differences with mean.
    double sqDiff = 0;
    for (int i = 0; i < n; i++)
        sqDiff += (a[i] - mean) * (a[i] - mean);
    return sqDiff / n;
}
 
double standardDeviation(vector<double> arr,
                         int n)
{
    return sqrt(variance(arr, n));
}
vector<int> smoothedZScore(vector<double> input)
{   
    //lag 5 for the smoothing functions
    int lag = 5;
    //3.5 standard deviations for signal
    float threshold = 3.5;
    //between 0 and 1, where 1 is normal influence, 0.5 is half
    float influence = .5;

    if (input.size() <= lag + 2)
    {
        vector<int> emptyVec;
        return emptyVec;
    }

    //Initialise variables
    vector<int> signals(input.size(), 0.0);
    vector<double> filteredY(input.size(), 0.0);
    vector<double> avgFilter(input.size(), 0.0);
    vector<double> stdFilter(input.size(), 0.0);
    vector<double> subVecStart(input.begin(), input.begin() + lag);
    avgFilter[lag] = findMean(subVecStart,subVecStart.size());
    stdFilter[lag] = standardDeviation(subVecStart,subVecStart.size());

    for (int i = lag + 1; i < input.size(); i++)
    {
        if (abs(input[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1])
        {
            if (input[i] > avgFilter[i - 1])
            {
                signals[i] = 1; //# Positive signal
            }
            else
            {
                signals[i] = -1; //# Negative signal
            }
            //Make influence lower
            filteredY[i] = influence* input[i] + (1 - influence) * filteredY[i - 1];
        }
        else
        {
            signals[i] = 0; //# No signal
            filteredY[i] = input[i];
        }
        //Adjust the filters
        vector<double> subVec(filteredY.begin() + i - lag, filteredY.begin() + i);
        avgFilter[i] = findMean(subVec,subVec.size());
        stdFilter[i] = standardDeviation(subVec,subVec.size());
    }
    return signals;
}

ifstream file;
int main(int argc, char* argv[]){
    char* filename = argv[1];
    file.open(filename);
    string line;
    getline(file,line);
    vector<string> tokens;
    stringstream tokenize(line);
    string temp;
    vector<double>v;
    while(getline(tokenize,temp,' ')) v.push_back(stod(temp));
    vector<int>Signals = smoothedZScore(v);
    int count = 0;
    for(int j = 0; j < Signals.size();j++){
        if(Signals[j]==1){ 
            count++;
            while(Signals[j]==1){
                j++;
            }
            j--;
        }
    }
    //cout << count << " " << v.size() << endl;
    int fps = 30;
    cout << count*60*fps/v.size() << endl;
    return 0;
}
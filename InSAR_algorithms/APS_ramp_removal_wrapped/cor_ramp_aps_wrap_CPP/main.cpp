#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <math.h>
#include <algorithm>

using namespace std;

// prototype
template <typename T>
vector<T> diff_mat(T *ar, const int col1, const int line1, const char order1, float th=0);
vector<float> retrieve_phase(float *ar2 , const int col2, const int line2);
vector<float> retrieve_amp(float *ar3 , const int col3, const int line3);
vector<float> mask_by_cor(float *ar4, float *ar5, const int ll, const float th_cor);
vector<float> phs_mag2rmg(float *ar6,float *ar7,const int ll2, const int col6);
vector<float> mask_by_M(float *ar8, float *ar9, const int ll1);
template <typename T>
T & wrapTo2Pi(T &number);

int main(int argc, char* argv[])
{  
    vector<float> f;
    vector<float> c;
    vector<short> dem;
    vector<float> ar_M;
    vector<float> p_all;
    vector<short> dem_all;
    struct stat result1;
    struct stat result2;
    struct stat result3;
    struct stat result4;
    
    
    string inFile = argv[1];
    string corFile = argv[2];
    string demFile = argv[3];
    string maskFile = argv[4];
    
    //vector<float> & getVectorByFilename(char &inFile, vector<float> &f);
    //for(int i=2280000;i<2283000;i++) { cout << f[i] << ","; }
    //cout << endl;
    

    stat(argv[1], &result1);
    int size1 = result1.st_size / 4;
    ifstream File1(inFile,ifstream::in|ifstream::binary) ;
    //File1.open (inFile, ios::in | ios::binary);
    stat(argv[2], &result2);
    int size2 = result2.st_size / 4;
    ifstream File2(corFile,ifstream::in|ifstream::binary);
    //File2.open (corFile, ios::in | ios::binary);
    stat(argv[3], &result3);
    int size3 = result3.st_size / 2;
    cout << "size3 is: " << size3 << endl;
    ifstream File3(demFile,ifstream::in|ifstream::binary);
    //File3.open (demFile, ios::in | ios::binary);
    stat(argv[4], &result4);
    int size4 = result1.st_size / 4;
    ifstream File4(maskFile,ifstream::in|ifstream::binary);
   
    
    int col = stoi(argv[5]);
    float th_d = atof(argv[6]);
    cout << "inital th_d is: " << th_d << endl;
    float th_coherence = stoi(argv[7]);
    int method = stoi(argv[8]);
    string outFile = argv[9];
    
    f.resize(size1);
    c.resize(size2);
    dem.resize(size3);
    ar_M.resize(size4);
    File1.read(reinterpret_cast<char*>(&f[0]), size1*4);
    File1.close();
    File2.read(reinterpret_cast<char*>(&c[0]),size2*4);
    File2.close();
    File3.read((char*)(&dem[0]),size3*sizeof(short));
    cout << "sizeof dem is: " << dem.size() << "sizeof f is: " << f.size() << endl;
    File3.close();
    File4.read(reinterpret_cast<char*>(&ar_M[0]),size4*4);
    File4.close();
    
    
    
    int S = f.size();
    int line = S / col / 2;
    int length_p = line*col;
    
    
    vector<float> ar_p = retrieve_phase(&*f.begin(),col,line);
    vector<float> ar_c = retrieve_phase(&*c.begin(),col,line);
    vector<float> ar_a = retrieve_amp(&*f.begin(),col,line);
    vector<float> ar_p1 = mask_by_cor(&*ar_p.begin(),&*ar_c.begin(),length_p,th_coherence);
    vector<float> ar_p2 = mask_by_M(&*ar_p1.begin(),&*ar_M.begin(),length_p);
    vector<float> ar_p2_double(ar_p2.begin(), ar_p2.end());

    
    vector<float> ar4_c = diff_mat(&*ar_p2_double.begin(),col,line,'C',th_d);
    cout << "th_d is: " << th_d << endl;
    cout << "Size of ar4_c is: " << ar4_c.size() << endl;
    vector<short> dem1_c = diff_mat(&*dem.begin(),col,line,'C',10000);
    cout << "Size of dem1_c is: " << dem1_c.size() << endl;
    vector<float> ar4_l = diff_mat(&*ar_p2_double.begin(),col,line,'L',th_d);
    cout << "Size of ar4_l is: " << ar4_l.size() << endl;
    vector<short> dem1_l = diff_mat(&*dem.begin(),col,line,'L',10000);
    cout << "Size of dem1_l is: " << dem1_l.size() << endl;
    p_all.reserve(ar4_c.size() + ar4_l.size() ); // preallocate memory    
    p_all.insert(p_all.end(), ar4_c.begin(), ar4_c.end() );
    p_all.insert( p_all.end(), ar4_l.begin(), ar4_l.end() );
    cout << "Size of p_all is: " << p_all.size() << endl;

    dem_all.reserve(dem1_c.size() + dem1_c.size()); // preallocate memory
    dem_all.insert(dem_all.end(), dem1_c.begin(), dem1_c.end() );
    dem_all.insert(dem_all.end(), dem1_l.begin(), dem1_l.end() );
    cout << "Size of dem_all is: " << dem_all.size() << endl;

    
    vector<float> temp;
    float sum_p = 0;
    int N = 0;
    float rate_elevation;
    
    for(int i=0;i<(p_all.size());i++)
    {
        if(fabs(dem_all[i])>20000 || isnan(dem_all[i]) || dem_all[i]==0)
        {
            p_all[i] = 0;
        }
        else
        {
            p_all[i] = p_all[i] / dem_all[i];
        }
        if (fabs(p_all[i])>0.000000000000001)
        {
            sum_p+=p_all[i];
            temp.push_back(p_all[i]);
            N+=1;
        }
    }
    
    if (method == 0)
    {
        rate_elevation = sum_p / N;
    }
    else if (method == 1)
    {
        int position = N / 2;
        sort(temp.begin(),temp.end());
        rate_elevation = temp[position];
    }
    else
    { 
        cout << "Method should be 0 or 1!" << endl;
        exit(1);
    }
    cout << "rate is:" << rate_elevation << endl;
    
    for(int i=0;i<ar_p.size();i++)
    {
        if(fabs(ar_p[i])<0.000000001)
        {
            ar_p[i] = 0;
        }
        else
        {
            float temp1 = rate_elevation*dem[i];
            temp1 = ar_p[i] - wrapTo2Pi(temp1);
            ar_p[i] = wrapTo2Pi(temp1);
        }
    }

    vector<float> new_data = phs_mag2rmg(&*ar_p.begin(),&*ar_a.begin(),length_p, col);
    cout << "size of new_data is: " << new_data.size() << endl;
    ofstream File5;
    const char* pointer = reinterpret_cast<char*>(&new_data[0]);
    File5.open(outFile,ios::out | ios::binary);
    File5.write(pointer, new_data.size()*4);
    File5.close();    
}

template <typename T>
vector<T> diff_mat(T *ar , const int col1, const int line1, const char order1, float th_d1)
{
    cout << "th is: " << th_d1 << endl;
    vector<vector<T> > A;
    vector<T> ar1;
    A.resize(line1);
    for(int k=0;k<line1;k++) { A[k].resize(col1); }
    for(int i=0;i<line1;i++) 
    {
        for(int j=0;j<col1;j++) 
        {
            A[i][j] = ar[col1*i+j];
        }
    }
    if (order1 == 'C')
    {
        int S = line1*(col1-1);
        ar1.resize(S);
        
        for(int ii=0;ii<line1;ii++)
        { 
            for(int jj=0;jj<col1-1;jj++)
            {
                if (fabs(ar1[(col1-1)*ii+jj]) > th_d1 || A[ii][jj+1]==0 || A[ii][jj]==0)
                {
                    ar1[(col1-1)*ii+jj] = 0;
                }
                else
                {
                    ar1[(col1-1)*ii+jj] = A[ii][jj+1] - A[ii][jj]; 
                } 
            }
        }
    }
    else
    {    
        int S = (line1-1)*col1;
        ar1.resize(S);
        for(int ii=0;ii<line1-1;ii++)
        {
            for(int jj=0;jj<col1;jj++)
            {
                if (ar1[col1*ii+jj] > th_d1 || A[ii+1][jj]==0 || A[ii][jj]==0)
                {
                    ar1[col1*ii+jj] = 0;
                }
                else
                {
                    ar1[col1*ii+jj] = A[ii+1][jj] - A[ii][jj];
                }
            }
        }   
    }
    return ar1;
}

vector<float> retrieve_phase(float *ar2 , const int col2, const int line2)
{
    vector<float> ar_phase;
    for(int i=0;i<(line2*2);i++)
    {
        if(i%2==1)
        {
            for(int j=0;j<col2;j++)
            {
                ar_phase.push_back(ar2[col2*i+j]);
            }
        }   
    }
    return ar_phase;
}

vector<float> retrieve_amp(float *ar3 , const int col3, const int line3)
{
    vector<float> ar_amp;
    for(int i=0;i<line3;i++)
    {
        if(i%2==0)
        {
            for(int j=0;j<col3;j++)
            {
                ar_amp.push_back(ar3[col3*i+j]);
            }
        }   
    }
    return ar_amp;
}

vector<float> mask_by_cor(float *ar4, float *ar5, const int ll, const float th_cor)
{
    vector<float> ar4_mask;
    ar4_mask.resize(ll);
    for(int i=0;i<ll;i++)
    {
        if (ar5[i]<th_cor)
        {
            ar4_mask[i] = 0;
        }
        else
        {
            ar4_mask[i] = ar4[i];
        }
    }
    return ar4_mask;
}

vector<float> mask_by_M(float *ar8, float *ar9, const int ll1)
{
    vector<float> ar8_mask;
    ar8_mask.resize(ll1);
    for(int i=0;i<ll1;i++)
    {
        if (ar9[i]==0)
        {
            ar8_mask[i] = 0;
        }
        else
        {
            ar8_mask[i] = ar8[i];
        }
    }
    return ar8_mask;
}

vector<float> phs_mag2rmg(float *ar6,float *ar7,const int ll2, const int col6)
{
    vector<float> new1;
    new1.resize(ll2*2);
    for(int i=0;i<(ll2*2);i++)
    {
        int j1 = i/col6;
        int j2 = i%col6;
        if (j1%2 == 0)
        {
            new1[i] = ar7[j1/2*col6+j2];
        }
        else
        {
            new1[i] = ar6[j1/2*col6+j2];
        }
    }
    return new1;
}

template<typename T>
vector<T> & getVectorByFilename(char *filename, vector<T> &vector) {
    struct stat fileStat;
    stat(filename, &fileStat);   //Get information of file
    long filesize = fileStat.st_size;//Get size of file 
    vector.resize(filesize/sizeof(T));//Assign space for vector
    ifstream file;
    file.open(filename, ifstream::in | ifstream::binary);

    //Code below uses large memory when dealing with big file
    file.read(reinterpret_cast<char*>(&vector[0]), filesize);//Read into vector
    file.close();//Close file
    return vector;
}

template <typename T>
T & wrapTo2Pi(T &number) {
    double pi = atan(1)*4;
    if(number>pi)
    {
        number = number + 2*pi*(floor(number/(2*pi)));
    }
    else if(number<(-1*pi))
    {
        number = number - 2*pi*floor(number/(2*pi));
    }
    number = (T) number;
    return number;
}
